"""Build the V595 evidence disposition from checked-in terminal artifacts.

This module does not run research code. It reads exact artifact paths and
recomputes small summaries from their rows. Missing or unsafe evidence stays
visible so a later reader can tell a resource block from a measured null.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.595"
INFERENCE_SUBSTRATE = "CPU aggregation from upstream artifacts, no LLM"
RANDOM_SEED = 6823
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_6823_v595_branch_disposition.json")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
TASK_IDS = tuple(f"exp{number}" for number in range(6810, 6824))
SOURCE_TASK_IDS = TASK_IDS[:-1]
DISPOSITION_ENUM = ("adopt", "narrow", "retire", "blocked")
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

TASK_PATHS = {
    "exp6810": Path("results/experiment_6810_v595_contract_manifest_preflight.json"),
    "exp6811": Path("results/experiment_6811_operational_obligation_automaton_v3.json"),
    "exp6812": Path("results/experiment_6812_sota_operational_handoff_corpus_v2.json"),
    "exp6813": Path("results/experiment_6813_selective_priority_arbiter_ab.json"),
    "exp6814": Path("results/experiment_6814_selective_priority_arbiter_cold_audit.json"),
    "exp6815": Path("results/experiment_6815_verified_memory_operation_stream.json"),
    "exp6816": Path("results/experiment_6816_residual_pressure_route_learning_ab.json"),
    "exp6817": Path("results/experiment_6817_route_memory_portability.json"),
    "exp6818": Path("results/experiment_6818_route_memory_global_cold_audit.json"),
    "exp6819": Path("results/experiment_6819_arc_stepwise_strategy_accrual.json"),
    "exp6820": Path("results/experiment_6820_arc_tool_gap_obligation_transport_v2.json"),
    "exp6821": Path("results/experiment_6821_arc_obligation_actions_to_progress_ab_v2.json"),
    "exp6822": Path("results/experiment_6822_arc_causal_adoption_audit.json"),
    "exp6823": RESULT_PATH,
}

OWNED_INPUT_PATHS = (
    DESIGN_PATH,
    ROADMAP_PATH,
    Path("openspec/capabilities/agentic-harness/spec.md"),
    Path("openspec/capabilities/constraint-verification/spec.md"),
    Path("openspec/capabilities/continuous-learning/spec.md"),
    REPORT_SPEC_PATH,
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
)

CONTRACT_OWNER_MAP = {
    "operational-obligation interface": {
        "spec_path": "openspec/capabilities/agentic-harness/spec.md",
        "requirement": "REQ-AGENTIC-6810-1",
        "implementing_task": "exp6811-operational-obligation-automaton-v3",
        "gate_field": "operational_automaton_fixture_ready",
    },
    "exact priority arbiter": {
        "spec_path": "openspec/capabilities/constraint-verification/spec.md",
        "requirement": "REQ-CONSTRAINT-6810",
        "implementing_task": "exp6813-selective-priority-arbiter-ab",
        "gate_field": "selective_arbiter_ab_completed",
    },
    "transactional verified memory": {
        "spec_path": "openspec/capabilities/continuous-learning/spec.md",
        "requirement": "REQ-CL-6810",
        "implementing_task": "exp6816-residual-pressure-route-learning-ab",
        "gate_field": "residual_route_learning_completed",
    },
    "live stepwise strategy path": {
        "spec_path": "openspec/capabilities/agentic-harness/spec.md",
        "requirement": "REQ-AGENTIC-6810-2",
        "implementing_task": "exp6819-arc-stepwise-strategy-accrual",
        "gate_field": "stepwise_strategy_accrual_ready",
    },
}

EXPECTED_GATES = {
    "exp6810": [],
    "exp6811": [("exp6810", "v595_contract_map_ready")],
    "exp6812": [("exp6811", "operational_automaton_fixture_ready")],
    "exp6813": [("exp6812", "operational_handoff_corpus_ready")],
    "exp6814": [("exp6813", "selective_arbiter_ab_completed")],
    "exp6815": [
        ("exp6812", "operational_handoff_corpus_ready"),
        ("exp6814", "selective_arbiter_audit_completed"),
    ],
    "exp6816": [("exp6815", "verified_memory_stream_ready")],
    "exp6817": [("exp6816", "residual_route_learning_completed")],
    "exp6818": [
        ("exp6816", "residual_route_learning_completed"),
        ("exp6817", "route_memory_portability_completed"),
    ],
    "exp6819": [("exp6811", "operational_automaton_fixture_ready")],
    "exp6820": [("exp6819", "stepwise_strategy_accrual_ready")],
    "exp6821": [
        ("exp6820", "tool_gap_obligation_transport_ready"),
        ("exp6814", "selective_arbiter_audit_completed"),
    ],
    "exp6822": [
        ("exp6819", "stepwise_strategy_accrual_ready"),
        ("exp6820", "tool_gap_obligation_transport_ready"),
        ("exp6821", "actions_to_progress_ab_completed"),
    ],
    "exp6823": [],
}

QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA_26 = "unsloth/gemma-4-26B-A4B-it-GGUF"
EXPECTED_RESEARCH_MODELS = {
    **{task_id: [] for task_id in TASK_IDS},
    "exp6812": [QWEN, GEMMA_26, GEMMA_31],
    "exp6819": [QWEN],
    "exp6820": [QWEN],
    "exp6821": [QWEN],
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "roadmap_identity",
    "contract_owner_map",
    "rows",
    "task_count",
    "terminal_class_counts",
    "gate_check_summary",
    "adversarial_findings",
    "row_recomputed_claims",
    "manifest_comparison",
    "selective_arbiter_disposition",
    "verified_route_memory_disposition",
    "live_arc_disposition",
    "disposition_enum",
    "solve_claim",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each field explains why it exists, so schema compliance remains meaningful.",
    "inference_substrate": "The substrate proves that this task only aggregates checked-in evidence.",
    "duration_s": "Measured wall time distinguishes a real audit from an invented duration.",
    "random_seed": "A fixed audit seed makes any ordered reduction repeatable.",
    "reproducibility_checksum": "One checksum binds the inputs, rows, commands, and output.",
    "source_artifact_hashes": "Exact hashes stop later file changes from silently changing this disposition.",
    "roadmap_identity": "The milestone and range prevent evidence from another plan entering this audit.",
    "contract_owner_map": "Existing specifications keep authority with the contract that owns it.",
    "rows": "Task and branch rows let a reader audit every inclusion and decision.",
    "task_count": "The fixed count prevents a missing task from disappearing from the denominator.",
    "terminal_class_counts": "Separate counts prevent blocked or flagged work from becoming success.",
    "gate_check_summary": "Expected and observed values make incomplete evidence diagnosable.",
    "adversarial_findings": "Validator findings remain visible and cannot be laundered away.",
    "row_recomputed_claims": "Headlines come from eligible source rows instead of copied prose.",
    "manifest_comparison": "Design differences remain explicit even though this synthesis is ungated.",
    "selective_arbiter_disposition": "The arbiter receives a decision independent of memory and ARC.",
    "verified_route_memory_disposition": "Memory causality and portability receive their own decision.",
    "live_arc_disposition": "Live progress receives a decision independent of transport readiness.",
    "disposition_enum": "A closed decision set prevents an ambiguous branch outcome.",
    "solve_claim": "False prevents this synthesis from becoming a game-level solve claim.",
    "verifier_is_oracle": "False keeps exact environment outcomes as the correctness authority.",
    "verdict_class": "The closed class states the evidence limit of the synthesis itself.",
    "honest_verdict": "A terminal and bounded sentence states what the available evidence supports.",
}

VERIFICATION_COMMANDS = (
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check python/carnot/experiment_6823_v595_branch_disposition.py tests/python/test_experiment_6823_v595_branch_disposition.py",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6823_v595_branch_disposition.json",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6823_v595_branch_disposition.json",
)


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes so hashes do not depend on formatting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str | None:
    """Hash one exact path, or retain absence as a null value."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _short_task_id(value: str) -> str:
    match = re.match(r"exp(\d+)", value)
    if not match:
        raise ValueError(f"invalid task id: {value}")
    return f"exp{match.group(1)}"


def parse_design(text: str) -> JsonDict:
    """Parse the four phase headings and exact deliverables from the design."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`?([0-9]{4}\.[0-9]{2}\.[0-9]+)`?", text)
    milestone = milestone_match.group(1) if milestone_match else None
    phase = 0
    tasks: list[JsonDict] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        phase_match = re.match(r"## Phase (\d+):", line)
        if phase_match:
            phase = int(phase_match.group(1))
            continue
        task_match = re.match(r"### Exp(\d+):\s*(.+)", line)
        if not task_match:
            continue
        deliverable = None
        for later in lines[index + 1 :]:
            found = re.match(r"\*\*Deliverable:\*\*\s*`([^`]+)`", later)
            if found:
                deliverable = found.group(1)
                break
            if later.startswith("### Exp") or later.startswith("## Phase"):
                break
        if deliverable is None:
            raise ValueError(f"design deliverable missing for Exp{task_match.group(1)}")
        tasks.append(
            {
                "task_id": f"exp{task_match.group(1)}",
                "title": task_match.group(2),
                "phase": phase,
                "deliverable": deliverable,
            }
        )
    return {
        "milestone": milestone,
        "phase_count": len(set(row["phase"] for row in tasks)),
        "tasks": tasks,
    }


def _research_models(prompt: str) -> list[str]:
    return sorted(model for model in (QWEN, GEMMA_26, GEMMA_31) if model in prompt)


def load_manifest(root: Path) -> list[JsonDict]:
    """Load the executed YAML while retaining gates, models, and deliverables."""

    payload = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("executed roadmap must contain a task list")
    rows = []
    for task in payload["tasks"]:
        if not isinstance(task, dict):
            continue
        task_id = _short_task_id(str(task.get("id", "")))
        gates = []
        for gate in task.get("gated_on") or []:
            gates.append(
                (_short_task_id(str(gate.get("upstream", ""))), gate.get("artifact_field"))
            )
        rows.append(
            {
                "task_id": task_id,
                "manifest_task_id": task.get("id"),
                "title": task.get("title"),
                "milestone": task.get("milestone"),
                "deliverable": task.get("deliverable"),
                "gates": gates,
                "agent_model": task.get("model"),
                "research_models": _research_models(str(task.get("prompt", ""))),
            }
        )
    return rows


def compare_manifest(
    design: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    observed_owner_map: Mapping[str, Any],
) -> JsonDict:
    """Compare every design surface without turning a mismatch into a gate."""

    differences: list[JsonDict] = []

    def check(field: str, expected: Any, observed: Any) -> None:
        if expected != observed:
            differences.append({"field": field, "expected": expected, "observed": observed})

    check("milestone", MILESTONE, design.get("milestone"))
    check("phase_count", 4, design.get("phase_count"))
    check(
        "design_task_ids", list(TASK_IDS), [row.get("task_id") for row in design.get("tasks", [])]
    )
    check("manifest_task_ids", list(TASK_IDS), [row.get("task_id") for row in manifest])
    design_by_id = {row["task_id"]: row for row in design.get("tasks", [])}
    manifest_by_id = {row["task_id"]: row for row in manifest}
    deliverables = []
    gates = []
    models = []
    for task_id in TASK_IDS:
        expected_path = TASK_PATHS[task_id].as_posix()
        design_path = design_by_id.get(task_id, {}).get("deliverable")
        manifest_path = manifest_by_id.get(task_id, {}).get("deliverable")
        deliverables.append(
            {
                "task_id": task_id,
                "expected": expected_path,
                "design": design_path,
                "executed": manifest_path,
                "matches": expected_path == design_path == manifest_path,
            }
        )
        check(f"{task_id}.deliverable", expected_path, manifest_path)
        observed_gates = manifest_by_id.get(task_id, {}).get("gates")
        gates.append(
            {
                "task_id": task_id,
                "expected": EXPECTED_GATES[task_id],
                "executed": observed_gates,
                "matches": EXPECTED_GATES[task_id] == observed_gates,
            }
        )
        check(f"{task_id}.gates", EXPECTED_GATES[task_id], observed_gates)
        observed_models = manifest_by_id.get(task_id, {}).get("research_models")
        expected_models = sorted(EXPECTED_RESEARCH_MODELS[task_id])
        models.append(
            {
                "task_id": task_id,
                "expected_research_models": expected_models,
                "executed_prompt_models": observed_models,
                "agent_model": manifest_by_id.get(task_id, {}).get("agent_model"),
                "matches": expected_models == observed_models,
            }
        )
        check(f"{task_id}.research_models", expected_models, observed_models)
    observed_owner_core = {
        contract: {field: observed_owner_map.get(contract, {}).get(field) for field in expected}
        for contract, expected in CONTRACT_OWNER_MAP.items()
    }
    check("contract_owner_map", CONTRACT_OWNER_MAP, observed_owner_core)
    return {
        "design_phase_count": design.get("phase_count"),
        "design_task_count": len(design.get("tasks", [])),
        "executed_task_count": len(manifest),
        "deliverables": deliverables,
        "gates": gates,
        "models": models,
        "owned_req_map_expected": CONTRACT_OWNER_MAP,
        "owned_req_map_observed": observed_owner_map,
        "differences": differences,
        "matches": not differences,
    }


def _missing_record(task_id: str, state: str, error: str) -> JsonDict:
    path = TASK_PATHS[task_id]
    return {
        "artifact_state": state,
        "path": path.as_posix(),
        "sha256": None,
        "payload": None,
        "source_verdict_class": state,
        "terminal_class": state,
        "eligible": False,
        "adversarial_report": None,
        "row_lint": None,
        "gate_check_summary": [
            {
                "check": f"{task_id}.exact_terminal_artifact",
                "expected": path.as_posix(),
                "observed": error,
            }
        ],
        "error": error,
    }


def load_source_artifacts(root: Path) -> dict[str, JsonDict]:
    """Load all exact paths and preserve every absent or malformed input."""

    records: dict[str, JsonDict] = {}
    for task_id in SOURCE_TASK_IDS:
        relative = TASK_PATHS[task_id]
        path = root / relative
        if not path.is_file():
            records[task_id] = _missing_record(task_id, "missing", "file_missing")
            continue
        file_hash = sha256_file(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            record = _missing_record(task_id, "malformed", f"{type(exc).__name__}: {exc}")
            record["sha256"] = file_hash
            records[task_id] = record
            continue
        verdict_class = payload.get("verdict_class") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or verdict_class not in CLOSED_VERDICT_CLASSES:
            record = _missing_record(task_id, "malformed", "top_level_or_verdict_class_invalid")
            record["sha256"] = file_hash
            records[task_id] = record
            continue
        declared_flagged = bool(payload.get("flagged_adversarial"))
        terminal_class = "flagged" if declared_flagged else verdict_class
        gate_summary = payload.get("gate_check_summary")
        records[task_id] = {
            "artifact_state": "present",
            "path": relative.as_posix(),
            "sha256": file_hash,
            "payload": payload,
            "source_verdict_class": verdict_class,
            "terminal_class": terminal_class,
            "eligible": verdict_class in {"positive", "null"} and not declared_flagged,
            "adversarial_report": None,
            "row_lint": None,
            "gate_check_summary": gate_summary if isinstance(gate_summary, list) else [],
            "error": None,
        }
    records["exp6823"] = {
        "artifact_state": "current_synthesis",
        "path": RESULT_PATH.as_posix(),
        "sha256": None,
        "payload": None,
        "source_verdict_class": "partial",
        "terminal_class": "current_synthesis",
        "eligible": False,
        "adversarial_report": None,
        "row_lint": None,
        "gate_check_summary": [],
        "error": None,
    }
    return records


def collect_source_hashes(root: Path) -> dict[str, str | None]:
    """Hash owned inputs and all thirteen upstream evidence paths."""

    paths = (*OWNED_INPUT_PATHS, *(TASK_PATHS[task] for task in SOURCE_TASK_IDS))
    return {path.as_posix(): sha256_file(root / path) for path in paths}


def apply_validators(
    root: Path,
    records: Mapping[str, JsonDict],
    adversarial: Callable[[Path], Mapping[str, Any]],
    row_lint: Callable[[Path], tuple[str, list[str]]],
) -> list[JsonDict]:
    """Run both validators and exclude any evidence with a reported concern."""

    findings: list[JsonDict] = []
    for task_id in SOURCE_TASK_IDS:
        record = records[task_id]
        if record["artifact_state"] != "present":
            continue
        path = root / record["path"]
        report = dict(adversarial(path))
        record["adversarial_report"] = report
        for flag in report.get("flags") or []:
            findings.append({"task_id": task_id, "validator": "adversarial", **flag})
        status, row_findings = row_lint(path)
        record["row_lint"] = {"status": status, "findings": row_findings}
        findings.extend(
            {"task_id": task_id, "validator": "row_verdict", "detail": detail}
            for detail in row_findings
        )
        if report.get("flag_count", 0) or status in {"findings", "unreadable"}:
            record["terminal_class"] = "flagged"
            record["eligible"] = False
    return findings


def _eligible_rows(records: Mapping[str, Mapping[str, Any]], task_id: str) -> list[JsonDict]:
    record = records.get(task_id, {})
    payload = record.get("payload")
    if not record.get("eligible") or not isinstance(payload, dict):
        return []
    rows = payload.get("rows")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _arm_rate(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, JsonDict]:
    values: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("arm"), str) and isinstance(row.get(field), bool):
            values[row["arm"]].append(row[field])
    return {
        arm: {"numerator": sum(items), "denominator": len(items), "rate": sum(items) / len(items)}
        for arm, items in values.items()
    }


def _arm_mean(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if (
            isinstance(row.get("arm"), str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            values[row["arm"]].append(float(value))
    return {arm: sum(items) / len(items) for arm, items in values.items()}


def _paired_delta(
    rows: Sequence[Mapping[str, Any]],
    left_arm: str,
    right_arm: str,
    field: str,
) -> JsonDict:
    pairs: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = row.get(field)
        if (
            isinstance(row.get("pair_id"), str)
            and row.get("arm") in {left_arm, right_arm}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            pairs[row["pair_id"]][row["arm"]] = float(value)
    deltas = [pair[left_arm] - pair[right_arm] for pair in pairs.values() if len(pair) == 2]
    if not deltas:
        return {"mean": None, "pair_count": 0, "cause": "no_eligible_paired_rows"}
    return {"mean": sum(deltas) / len(deltas), "pair_count": len(deltas), "cause": None}


def recompute_selective(records: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Reduce held arbiter rows while separating safety from utility."""

    source_rows = _eligible_rows(records, "exp6813")
    rows = [row for row in source_rows if row.get("split") == "held"]
    if not rows and source_rows and all("split" not in row for row in source_rows):
        rows = source_rows
    progress = _paired_delta(rows, "selective_priority", "flat_reject_retry", "accepted_progress")
    retries = _paired_delta(rows, "flat_reject_retry", "selective_priority", "retry_count")
    safety = _arm_rate(rows, "accepted_hard_violation")
    false_intervention = _arm_rate(
        [row for row in rows if row.get("base_already_valid") is True], "false_intervention"
    )
    harmful = _arm_rate(rows, "harmful_selection")
    return {
        "eligible_row_count": len(rows),
        "hard_safety": safety,
        "utility": {
            "accepted_progress_by_arm": _arm_mean(rows, "accepted_progress"),
            "retry_cost_by_arm": _arm_mean(rows, "retry_count"),
            "paired_progress_delta": progress,
            "flat_minus_selective_retry_delta": retries,
        },
        "false_intervention": false_intervention,
        "harmful_selection": harmful,
        "positive_gate": bool(
            rows
            and safety.get("selective_priority", {}).get("numerator") == 0
            and harmful.get("selective_priority", {}).get("numerator") == 0
            and (
                (progress["mean"] is not None and progress["mean"] > 0)
                or (retries["mean"] is not None and retries["mean"] > 0)
            )
        ),
    }


def _named_difference(means: Mapping[str, float], left: str, right: str) -> float | None:
    if left not in means or right not in means:
        return None
    return means[left] - means[right]


def recompute_memory(records: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Reduce local utility, causal writes, safety, and portability separately."""

    rows = _eligible_rows(records, "exp6816")
    portability_rows = _eligible_rows(records, "exp6817")
    means = _arm_mean(rows, "exact_utility")
    portability_means = _arm_mean(portability_rows, "exact_utility")
    credited = sum(row.get("credited_write") is True for row in rows)
    harm_fields = ("support_harm", "retention_harm", "hard_case_harm")
    harm_count = sum(row.get(field) is True for row in rows for field in harm_fields)
    utility_delta = _named_difference(means, "residual_pressure", "frozen_memory")
    portability_delta = _named_difference(portability_means, "residual_pressure", "frozen_memory")
    return {
        "eligible_row_count": len(rows),
        "utility": {"mean_by_arm": means, "residual_minus_frozen": utility_delta},
        "causality": {"credited_writes": credited},
        "safety": {"harm_count": harm_count, "fields": list(harm_fields)},
        "portability": {
            "eligible_row_count": len(portability_rows),
            "mean_by_arm": portability_means,
            "residual_minus_frozen": portability_delta,
        },
        "positive_gate": bool(
            utility_delta is not None
            and utility_delta > 0
            and portability_delta is not None
            and portability_delta > 0
            and credited > 0
            and harm_count == 0
        ),
    }


def recompute_live_arc(records: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Keep transport readiness separate from measured live progress."""

    transport_record = records.get("exp6820", {})
    transport_payload = transport_record.get("payload")
    transport_ready = (
        transport_payload.get("tool_gap_obligation_transport_ready")
        if transport_record.get("eligible") and isinstance(transport_payload, dict)
        else None
    )
    rows = _eligible_rows(records, "exp6821")
    actions_delta = _paired_delta(rows, "control_unset", "obligation_routed", "actions_to_progress")
    hard_count = sum(
        row.get("hard_violation") is True or row.get("accepted_hard_violation") is True
        for row in rows
    )
    false_interventions = sum(row.get("false_intervention") is True for row in rows)
    return {
        "transport": {
            "ready": transport_ready,
            "eligible_row_count": len(_eligible_rows(records, "exp6820")),
        },
        "progress": {
            "eligible_row_count": len(rows),
            "control_minus_treatment_actions": actions_delta["mean"],
            "pair_count": actions_delta["pair_count"],
            "cause": actions_delta["cause"],
        },
        "hard_safety": {
            "hard_violation_count": hard_count,
            "false_intervention_count": false_interventions,
        },
        "solve_claim": False,
        "positive_gate": bool(
            transport_ready is True
            and actions_delta["mean"] is not None
            and actions_delta["mean"] > 0
            and hard_count == 0
        ),
    }


def recompute_claims(records: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Return the three independent branch reductions without pooling."""

    return {
        "selective_arbiter": recompute_selective(records),
        "verified_route_memory": recompute_memory(records),
        "live_arc": recompute_live_arc(records),
    }


def decide_disposition(
    branch: str,
    records: Mapping[str, Mapping[str, Any]],
    required_tasks: Sequence[str],
    *,
    positive_gate: bool,
) -> JsonDict:
    """Close one branch only after all required evidence is independently eligible."""

    failed = []
    classes = []
    for task_id in required_tasks:
        record = records.get(task_id, {})
        terminal = record.get("terminal_class", "missing")
        source_class = record.get("source_verdict_class", terminal)
        classes.append(source_class)
        if not record.get("eligible"):
            failed.append(
                {
                    "check": f"{branch}.{task_id}.eligible_terminal_evidence",
                    "expected": "eligible positive or null terminal evidence",
                    "observed": terminal,
                }
            )
    if failed:
        disposition = "blocked"
    elif positive_gate and all(value == "positive" for value in classes):
        disposition = "adopt"
    elif classes and all(value == "null" for value in classes):
        disposition = "retire"
    else:
        disposition = "narrow"
    return {
        "branch": branch,
        "disposition": disposition,
        "required_tasks": list(required_tasks),
        "source_verdict_classes": classes,
        "positive_gate_recomputed": positive_gate,
        "independent_cold_audit_required": True,
        "failed_checks": failed,
        "next_action": {
            "adopt": "Adopt only the audited branch behavior and keep exact authorities unchanged.",
            "narrow": "Keep the supported boundary and remove the unsupported positive claim.",
            "retire": "Retire the tested mechanism after its complete clean null.",
            "blocked": "Keep the branch blocked until the named terminal evidence exists.",
        }[disposition],
    }


def build_dispositions(
    records: Mapping[str, Mapping[str, Any]], claims: Mapping[str, Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Issue the three branch decisions with their own producer and cold audit."""

    return {
        "selective_arbiter": decide_disposition(
            "selective_arbiter",
            records,
            ("exp6813", "exp6814"),
            positive_gate=bool(claims["selective_arbiter"]["positive_gate"]),
        ),
        "verified_route_memory": decide_disposition(
            "verified_route_memory",
            records,
            ("exp6816", "exp6817", "exp6818"),
            positive_gate=bool(claims["verified_route_memory"]["positive_gate"]),
        ),
        "live_arc": decide_disposition(
            "live_arc",
            records,
            ("exp6819", "exp6820", "exp6821", "exp6822"),
            positive_gate=bool(claims["live_arc"]["positive_gate"]),
        ),
    }


def _task_rows(
    manifest: Sequence[Mapping[str, Any]], records: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    manifest_by_id = {row["task_id"]: row for row in manifest}
    rows = []
    for task_id in TASK_IDS:
        record = records.get(task_id)
        if record is None and task_id == "exp6823":
            record = {
                "artifact_state": "current_synthesis",
                "sha256": None,
                "payload": None,
                "source_verdict_class": "partial",
                "terminal_class": "current_synthesis",
                "eligible": False,
                "gate_check_summary": [],
                "adversarial_report": None,
                "row_lint": None,
            }
        if record is None:
            raise ValueError(f"source record missing for {task_id}")
        payload = record.get("payload")
        source_rows = payload.get("rows") if isinstance(payload, dict) else None
        terminal = record["terminal_class"]
        rows.append(
            {
                "row_type": "task",
                "task_id": task_id,
                "manifest_task_id": manifest_by_id.get(task_id, {}).get("manifest_task_id"),
                "path": TASK_PATHS[task_id].as_posix(),
                "source_sha256": record.get("sha256"),
                "artifact_state": record["artifact_state"],
                "source_verdict_class": record["source_verdict_class"],
                "terminal_class": terminal,
                "eligible_for_positive_claim": bool(record.get("eligible")),
                "exclusion_reason": None
                if record.get("eligible")
                else f"terminal_class={terminal}",
                "source_row_count": len(source_rows) if isinstance(source_rows, list) else 0,
                "gate_check_summary": record.get("gate_check_summary") or [],
                "adversarial_report": record.get("adversarial_report"),
                "row_verdict_report": record.get("row_lint"),
            }
        )
    return rows


def _gate_summary(
    task_rows: Sequence[Mapping[str, Any]], manifest_comparison: Mapping[str, Any]
) -> list[JsonDict]:
    failures = []
    for row in task_rows:
        if row.get("terminal_class") in {
            "missing",
            "malformed",
            "blocked",
            "disqualified",
            "partial",
            "flagged",
        }:
            source_gates = row.get("gate_check_summary") or []
            if source_gates:
                failures.extend({"task_id": row["task_id"], **gate} for gate in source_gates)
            else:
                failures.append(
                    {
                        "task_id": row["task_id"],
                        "check": "eligible_terminal_evidence",
                        "expected": "positive or null clean terminal evidence",
                        "observed": row.get("terminal_class"),
                    }
                )
    failures.extend(
        {
            "task_id": "manifest",
            "check": difference["field"],
            "expected": difference["expected"],
            "observed": difference["observed"],
        }
        for difference in manifest_comparison["differences"]
    )
    return failures


def _checksum_view(artifact: Mapping[str, Any]) -> JsonDict:
    view = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    view["verification_commands"] = list(VERIFICATION_COMMANDS)
    return view


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind all stable output and the commands used to verify it."""

    return f"sha256:{hashlib.sha256(canonical_json(_checksum_view(artifact))).hexdigest()}"


def assemble_artifact(
    *,
    run_date: str,
    duration_s: float,
    manifest: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    adversarial_findings: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Assemble one stable artifact from already loaded and checked evidence."""

    owner_payload = records.get("exp6810", {}).get("payload")
    observed_owners = (
        owner_payload.get("contract_owner_map")
        if isinstance(owner_payload, dict)
        and isinstance(owner_payload.get("contract_owner_map"), dict)
        else CONTRACT_OWNER_MAP
    )
    manifest_comparison = compare_manifest(design, manifest, observed_owners)
    claims = recompute_claims(records)
    decisions = build_dispositions(records, claims)
    task_rows = _task_rows(manifest, records)
    branch_rows = [
        {"row_type": "branch", **decisions[name]}
        for name in (
            "selective_arbiter",
            "verified_route_memory",
            "live_arc",
        )
    ]
    counts = dict(sorted(Counter(row["terminal_class"] for row in task_rows).items()))
    blocked = any(row["disposition"] == "blocked" for row in decisions.values())
    artifact: JsonDict = {
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "roadmap_identity": {
            "milestone": MILESTONE,
            "first_experiment": 6810,
            "last_experiment": 6823,
            "execution_date": run_date,
            "design_sha256": source_hashes.get(DESIGN_PATH.as_posix()),
            "executed_roadmap_sha256": source_hashes.get(ROADMAP_PATH.as_posix()),
        },
        "contract_owner_map": observed_owners,
        "rows": [*task_rows, *branch_rows],
        "task_count": len(task_rows),
        "terminal_class_counts": counts,
        "gate_check_summary": _gate_summary(task_rows, manifest_comparison),
        "adversarial_findings": list(adversarial_findings),
        "row_recomputed_claims": claims,
        "manifest_comparison": manifest_comparison,
        "selective_arbiter_disposition": decisions["selective_arbiter"],
        "verified_route_memory_disposition": decisions["verified_route_memory"],
        "live_arc_disposition": decisions["live_arc"],
        "disposition_enum": list(DISPOSITION_ENUM),
        "solve_claim": False,
        "verifier_is_oracle": False,
        "verdict_class": "partial" if blocked else "null",
        "honest_verdict": (
            "complete_partial: V595 preserved every available terminal row; one or more "
            "branches remain blocked by named missing or excluded independent evidence."
            if blocked
            else "complete: V595 branch decisions are closed and no new scientific claim is made."
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay the compact output contract before the atomic write."""

    errors = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("artifact fields do not match the required schema")
    if artifact.get("task_count") != 14:
        errors.append("task_count must equal fourteen")
    rows = artifact.get("rows") if isinstance(artifact.get("rows"), list) else []
    task_rows = [row for row in rows if isinstance(row, dict) and row.get("row_type") == "task"]
    replayed = dict(sorted(Counter(row.get("terminal_class") for row in task_rows).items()))
    if replayed != artifact.get("terminal_class_counts"):
        errors.append("terminal class counts do not replay")
    if any(
        row.get("eligible_for_positive_claim") is True
        and row.get("terminal_class") not in {"positive", "null"}
        for row in task_rows
    ):
        errors.append("excluded task row is positive-eligible")
    for field in (
        "selective_arbiter_disposition",
        "verified_route_memory_disposition",
        "live_arc_disposition",
    ):
        value = artifact.get(field)
        if not isinstance(value, dict) or value.get("disposition") not in DISPOSITION_ENUM:
            errors.append(f"{field} has an invalid decision")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def atomic_write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the result only after a complete JSON document exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_validator(root: Path, relative: Path, function_name: str) -> Callable[..., Any]:
    """Load a repository validator from the exact source path that was hashed."""

    module_name = f"_exp6823_{relative.stem}"
    spec = importlib.util.spec_from_file_location(module_name, root / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def run(root: Path, run_date: str) -> JsonDict:
    """Read, validate, reduce, and publish the V595 disposition."""

    started = time.perf_counter()
    design = parse_design((root / DESIGN_PATH).read_text(encoding="utf-8"))
    manifest = load_manifest(root)
    records = load_source_artifacts(root)
    verify_artifact = _load_validator(root, ADVERSARIAL_PATH, "verify_artifact")
    check_artifact = _load_validator(root, ROW_LINT_PATH, "check_artifact")

    findings = apply_validators(root, records, verify_artifact, check_artifact)
    artifact = assemble_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        manifest=manifest,
        design=design,
        records=records,
        source_hashes=collect_source_hashes(root),
        adversarial_findings=findings,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    atomic_write_json(root / RESULT_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the required date and run the deterministic local synthesis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    artifact = run(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "result": RESULT_PATH.as_posix(),
                "verdict_class": artifact["verdict_class"],
                "dispositions": {
                    "selective_arbiter": artifact["selective_arbiter_disposition"]["disposition"],
                    "verified_route_memory": artifact["verified_route_memory_disposition"][
                        "disposition"
                    ],
                    "live_arc": artifact["live_arc_disposition"]["disposition"],
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
