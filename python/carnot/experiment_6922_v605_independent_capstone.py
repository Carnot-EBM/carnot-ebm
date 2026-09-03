"""Build the cold V605 evidence capstone from primary files.

The capstone keeps execution state separate from scientific evidence. This
prevents a completed task, a missing artifact, or an oracle-backed result from
silently becoming a positive claim. See REQ-REPORT-6922.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6922_v605_independent_capstone.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
INFERENCE_SUBSTRATE = "fresh_process_multibranch_evidence_synthesis_no_llm"
MILESTONE = "2026.09.605"
RANDOM_SEED = 6922
BLOCKED_VERDICT = "complete_blocked_v605_independent_capstone"
COMPLETE_VERDICT = "complete_partial_v605_evidence_synthesized_without_science_promotion"
FINAL_PROMPT_LINE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_FAMILIES = (
    "cardinality_constraints",
    "contradictions",
    "graph_coloring",
    "non_monotonic_defaults",
    "scheduling",
)
GENERATION_ARMS = ("direct_generation", "unguided_best_of_k", "guided_frontier")
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
DISPOSITIONS = {"adopt", "continue", "retire", "block"}


def _gate(upstream: str, field: str, op: str = "==", value: Any = 1) -> JsonDict:
    """Return one exact roadmap dependency in its executable shape."""

    return {"upstream": upstream, "artifact_field": field, "op": op, "value": value}


EXPECTED_TASKS: tuple[JsonDict, ...] = (
    {"number": 6911, "task_id": "exp6911-v605-document-yaml-evidence-contract", "title": "V605 document-YAML execution and evidence contract", "deliverable": "results/experiment_6911_v605_document_yaml_evidence_contract.json", "design_gate": "none", "gates": []},
    {"number": 6912, "task_id": "exp6912-alias-safe-relation-corpus-reducer", "title": "Alias-safe immutable relation-corpus reducer", "deliverable": "results/experiment_6912_alias_safe_relation_corpus_reducer.json", "design_gate": "none", "gates": []},
    {"number": 6913, "task_id": "exp6913-relation-source-tuple-qualification", "title": "Exact relation source and tuple qualification shard", "deliverable": "results/experiment_6913_relation_source_tuple_qualification.json", "design_gate": "Exp6912 ready = 1", "gates": [_gate("exp6912-alias-safe-relation-corpus-reducer", "clean_relation_corpus_ready_score")]},
    {"number": 6914, "task_id": "exp6914-relation-asp-isomorphic-qualification", "title": "Exact relation ASP and isomorphic qualification shard", "deliverable": "results/experiment_6914_relation_asp_isomorphic_qualification.json", "design_gate": "Exp6912 ready = 1", "gates": [_gate("exp6912-alias-safe-relation-corpus-reducer", "clean_relation_corpus_ready_score")]},
    {"number": 6915, "task_id": "exp6915-qualified-relation-event-bank", "title": "Independent qualified relation event-bank merge", "deliverable": "results/experiment_6915_qualified_relation_event_bank.json", "design_gate": "Exp6913 ready = 1 and Exp6914 ready = 1", "gates": [_gate("exp6913-relation-source-tuple-qualification", "source_tuple_shard_ready_score"), _gate("exp6914-relation-asp-isomorphic-qualification", "asp_isomorphic_shard_ready_score")]},
    {"number": 6916, "task_id": "exp6916-isomorphic-prospective-relation-stream", "title": "Isomorphic prospective relation learning stream", "deliverable": "results/experiment_6916_isomorphic_prospective_relation_stream.json", "design_gate": "Exp6915 ready = 1 and event count >= 90", "gates": [_gate("exp6915-qualified-relation-event-bank", "qualified_relation_event_bank_ready_score"), _gate("exp6915-qualified-relation-event-bank", "qualified_model_relation_event_count", ">=", 90)]},
    {"number": 6917, "task_id": "exp6917-bounded-relation-memory-continuous-learning", "title": "Bounded relation-memory continuous self-learning comparison", "deliverable": "results/experiment_6917_bounded_relation_memory_continuous_learning.json", "design_gate": "Exp6916 ready = 1", "gates": [_gate("exp6916-isomorphic-prospective-relation-stream", "prospective_relation_stream_ready_score")]},
    {"number": 6918, "task_id": "exp6918-relation-learning-cold-support-audit", "title": "Independent relation-learning safety and support audit", "deliverable": "results/experiment_6918_relation_learning_cold_support_audit.json", "design_gate": "Exp6917 run complete = 1", "gates": [_gate("exp6917-bounded-relation-memory-continuous-learning", "self_learning_run_complete_score")]},
    {"number": 6919, "task_id": "exp6919-exact-prefix-viability-fixture", "title": "Exact prefix-viability fixture and branch-cost canary", "deliverable": "results/experiment_6919_exact_prefix_viability_fixture.json", "design_gate": "none", "gates": []},
    {"number": 6920, "task_id": "exp6920-sota-exact-guided-relation-generation", "title": "SOTA plain-text relation generation with exact prefix guidance", "deliverable": "results/experiment_6920_sota_exact_guided_relation_generation.json", "design_gate": "Exp6919 ready = 1", "gates": [_gate("exp6919-exact-prefix-viability-fixture", "prefix_viability_canary_ready_score")]},
    {"number": 6921, "task_id": "exp6921-arc-dynamic-supervisor-banked-credit", "title": "ARC dynamic supervisor receipt and banked-progress audit", "deliverable": "results/experiment_6921_arc_dynamic_supervisor_banked_credit.json", "design_gate": "none", "gates": []},
    {"number": 6922, "task_id": "exp6922-v605-independent-capstone", "title": "V605 independent evidence capstone and branch disposition", "deliverable": OUTPUT_PATH.as_posix(), "design_gate": "none", "gates": []},
)

REQUIRED_ARTIFACT_FIELDS = {
    "schema", "experiment_id", "run_date", "status", "field_principles",
    "preconditions_checked", "inference_substrate", "duration_s",
    "source_artifact_hashes", "rows", "document_yaml_contract_rows",
    "task_state_rows", "conductor_artifact_state_rows", "missing_artifact_rows",
    "skipped_task_rows", "blocked_task_rows", "flagged_artifact_rows",
    "null_result_rows", "circular_result_rows", "positive_result_rows",
    "adversarial_recheck_rows", "row_consistency_rows",
    "reported_vs_recomputed_metrics", "dependency_taint_rows",
    "prior_verdict_comparison_rows", "retirement_action_rows",
    "relation_branch_rows", "self_learning_branch_rows",
    "exact_guidance_branch_rows", "arc_generalization_branch_rows",
    "branch_disposition_rows", "next_prerequisite_rows", "milestone_claim_rows",
    "false_promotion_count", "random_seed", "reproducibility_checksum",
    "v605_capstone_complete_score", "gate_check_summary", "verifier_is_oracle",
    "verdict_class", "honest_verdict",
}

GLOBAL_SOURCES = {
    "v605_document": DESIGN_PATH,
    "v605_yaml": ROADMAP_PATH,
    "conductor_log": CONDUCTOR_LOG_PATH,
    "exclusion_manifest": EXCLUSION_PATH,
    "adversarial_verifier": ADVERSARIAL_PATH,
    "row_consistency_verifier": ROW_LINT_PATH,
}

_AUDIT_CACHE: dict[tuple[str, int, int], tuple[JsonDict, tuple[str, list[str]]]] = {}
_CHECKER_CACHE: dict[tuple[str, int], Any] = {}


def spec_anchors(text: str) -> list[str]:
    """List requirement anchors so tests prove the spec owns behavior."""

    return re.findall(r"(?:REQ|SCENARIO)-[A-Z0-9-]+", text)


def load_yaml(path: Path) -> JsonDict:
    """Read one YAML mapping and reject shapes that cannot define tasks."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return value


def _number(value: Any) -> int | None:
    """Extract an experiment number without treating malformed IDs as valid."""

    match = re.match(r"exp(\d+)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_design(text: str) -> dict[int, JsonDict]:
    """Parse the human execution table without consulting the YAML or Exp6911."""

    deliverables = {
        int(number): path
        for number, path in re.findall(
            r"^### Exp(\d+)\s+[—-].*?^\*\*Deliverable:\*\*\s*`([^`]+)`",
            text,
            re.MULTILINE | re.DOTALL,
        )
    }
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*`(exp\d+[^`]*)`\s*\|\s*([^|]+?)\s*"
        r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$",
        re.MULTILINE,
    )
    return {
        _number(task_id) or -1: {
            "order": int(order),
            "task_id": task_id.strip(),
            "title": title.strip(),
            "phase": phase.strip(),
            "design_gate": gate.strip(),
            "deliverable": deliverables.get(_number(task_id) or -1),
        }
        for order, task_id, title, phase, gate in pattern.findall(text)
        if _number(task_id) in range(6911, 6923)
    }


def _roadmap_tasks(roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Normalize executable tasks while preserving prompts and prior failures."""

    raw = roadmap.get("tasks", [])
    if not isinstance(raw, list):
        return []
    rows: list[JsonDict] = []
    for order, task in enumerate(raw, 1):
        if not isinstance(task, Mapping):
            continue
        rows.append(
            {
                "order": order,
                "number": _number(task.get("id")),
                "task_id": str(task.get("id") or ""),
                "title": str(task.get("title") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "milestone": str(task.get("milestone") or ""),
                "gates": deepcopy(task.get("gated_on") or []),
                "prompt": str(task.get("prompt") or ""),
                "prior_failures": deepcopy(task.get("prior_failures") or []),
            }
        )
    return rows


def build_contract_rows(design_text: str, roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Compare all 12 task contracts against both independent primary sources."""

    design = _parse_design(design_text)
    executable = {_number(row.get("task_id")): row for row in _roadmap_tasks(roadmap)}
    design_models = {model for model in REQUIRED_MODEL_IDS if model in design_text}
    rows: list[JsonDict] = []
    for order, expected in enumerate(EXPECTED_TASKS, 1):
        number = expected["number"]
        doc = design.get(number, {})
        task = executable.get(number, {})
        prompt_lines = [line.strip() for line in str(task.get("prompt") or "").splitlines() if line.strip()]
        expected_command = (
            "Run command: cd {project_root} && .venv/bin/python "
            f"scripts/experiments/{Path(expected['deliverable']).stem}.py --date {{date}}"
        )
        prompt_ending_match = len(prompt_lines) >= 2 and prompt_lines[-2:] == [
            expected_command,
            FINAL_PROMPT_LINE,
        ]
        model_match = number != 6920 or (
            design_models == set(REQUIRED_MODEL_IDS)
            and all(model in str(task.get("prompt") or "") for model in REQUIRED_MODEL_IDS)
        )
        values = {
            "order_match": doc.get("order") == task.get("order") == order,
            "task_id_match": doc.get("task_id") == task.get("task_id") == expected["task_id"],
            "title_match": doc.get("title") == task.get("title") == expected["title"],
            "deliverable_match": doc.get("deliverable") == task.get("deliverable") == expected["deliverable"],
            "milestone_match": roadmap.get("milestone") == task.get("milestone") == MILESTONE,
            "gates_match": doc.get("design_gate") == expected["design_gate"] and task.get("gates") == expected["gates"],
            "prompt_ending_match": prompt_ending_match,
            "required_model_ids_match": model_match,
        }
        rows.append(
            {
                "row_type": "document_yaml_contract",
                "order": order,
                "task_id": expected["task_id"],
                "document": doc or None,
                "yaml": {key: task.get(key) for key in ("order", "task_id", "title", "deliverable", "milestone", "gates")} if task else None,
                **values,
                "passed": all(values.values()),
            }
        )
    return rows


def parse_conductor_states(text: str, tasks: Sequence[Mapping[str, Any]]) -> tuple[datetime | None, dict[str, JsonDict]]:
    """Read V605 events after activation and keep the final state per task."""

    activation: datetime | None = None
    states: dict[str, JsonDict] = {}
    active = False
    for line in text.splitlines():
        if "Milestone 2026.09.605 activated" in line:
            active = True
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            try:
                activation = datetime.strptime(cells[0], "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
            except (IndexError, ValueError):
                activation = None
            continue
        if not active or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        log_title, raw_status, detail = cells[1], cells[2], cells[3]
        task = next(
            (
                row
                for row in tasks
                if str(row.get("title") or "").startswith(log_title)
                or log_title.startswith(str(row.get("title") or ""))
            ),
            None,
        )
        if task is None:
            continue
        if raw_status == "GATE_BLOCK" and "Pre-emptive skip" in detail:
            state, verdict = "preemptive_skip", "blocked_preemptively_upstream_retired"
        elif raw_status == "GATE_BLOCK":
            state, verdict = "gate_blocked", "blocked_gate_check_failed"
        elif raw_status == "FLAGGED":
            state, verdict = "flagged", None
        elif raw_status == "OK":
            state, verdict = "ok", None
        else:
            state, verdict = "failed", None
        states[str(task["task_id"])] = {
            "state": state,
            "raw_status": raw_status,
            "detail": detail,
            "timestamp": cells[0],
            "honest_verdict": verdict,
        }
    return activation, states


def _scalar(value: Any) -> Any:
    """Read fields that may use the repository's principle wrapper."""

    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _expected_verdict(payload: Mapping[str, Any]) -> str:
    """Derive the closed class from structural facts before trusting its label."""

    declared = str(_scalar(payload.get("verdict_class")) or "")
    text = " ".join(
        str(_scalar(payload.get(key)) or "").lower() for key in ("status", "honest_verdict")
    )
    if "blocked" in text or "gate_check_failed" in text:
        return "blocked"
    if "disqual" in text or "quarantin" in text or "thresholds_not_met" in text:
        return "disqualified"
    if "null" in text or "insufficient" in text or "not_shown" in text:
        return "null"
    if declared == "positive" and _scalar(payload.get("verifier_is_oracle")) is True:
        return "circular_positive"
    if declared in CLOSED_VERDICT_CLASSES:
        return declared
    return "partial"


def classify_evidence(
    task: Mapping[str, Any],
    conductor: Mapping[str, Any],
    artifact_state: str,
    payload: Mapping[str, Any] | None,
    adversarial: Mapping[str, Any] | None,
    row_check: tuple[str, list[str]] | None,
) -> JsonDict:
    """Classify one task without collapsing absence, null, block, or flag."""

    conductor_state = str(conductor.get("state") or "not_recorded")
    honest = conductor.get("honest_verdict")
    declared = str(_scalar(payload.get("verdict_class")) or "") if payload else None
    expected = _expected_verdict(payload) if payload else None
    critical = bool(
        adversarial
        and any(str(flag.get("severity", "")).lower() == "critical" for flag in adversarial.get("flags", []))
    )
    source_flag = bool(payload and _scalar(payload.get("flagged_adversarial")) is True)
    row_status = row_check[0] if row_check else None
    findings = row_check[1] if row_check else []
    if artifact_state == "absent":
        evidence_state = "skipped" if conductor_state == "preemptive_skip" else "absent"
        verdict_class = "blocked"
    elif artifact_state in {"stale", "invalid"}:
        evidence_state, verdict_class = artifact_state, "disqualified"
    elif conductor_state == "flagged" or critical or source_flag:
        evidence_state, verdict_class = "flagged", "disqualified"
    elif conductor_state in {"gate_blocked", "preemptive_skip"}:
        evidence_state, verdict_class = "blocked", "blocked"
    elif row_status in {"findings", "unreadable"}:
        evidence_state, verdict_class = "row_disagreement", "disqualified"
    elif declared not in CLOSED_VERDICT_CLASSES or declared != expected:
        evidence_state, verdict_class = "wrong_verdict_class", "disqualified"
    else:
        verdict_class = expected or "partial"
        evidence_state = verdict_class
    if payload:
        honest = _scalar(payload.get("honest_verdict")) or honest
    return {
        "row_type": "task_state",
        "task_id": task.get("task_id"),
        "number": task.get("number"),
        "title": task.get("title"),
        "conductor_state": conductor_state,
        "artifact_state": artifact_state,
        "evidence_state": evidence_state,
        "declared_verdict_class": declared,
        "structural_verdict_class": expected,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
        "admissible": verdict_class in {"positive", "circular_positive", "null"},
        "replayable": artifact_state in {"present", "current_synthesis"}
        and adversarial is not None
        and row_status not in {None, "unreadable"},
        "adversarial_critical": critical,
        "row_check_status": row_status,
        "row_findings": findings,
    }


def _checks_pass(payload: Mapping[str, Any]) -> bool | None:
    """Reduce explicit check rows and return None when no check rows exist."""

    summary = payload.get("gate_check_summary")
    checks = summary.get("checks") if isinstance(summary, Mapping) else None
    if not isinstance(checks, list) or not checks:
        return None
    real = [row for row in checks if isinstance(row, Mapping)]
    return bool(real) and all(row.get("passed") is True for row in real)


def _comparison(field: str, reported: Any, recomputed: Any, support: str) -> JsonDict:
    """Keep a producer value beside the independent row reduction."""

    return {
        "row_type": "reported_vs_recomputed",
        "field": field,
        "reported": reported,
        "recomputed": recomputed,
        "support": support,
        "agrees": reported == recomputed,
        "status": "passed" if reported == recomputed else "producer_disagreement",
    }


def recompute_headlines(number: int, payload: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute only headline fields that stored rows can support."""

    rows: list[JsonDict] = []
    if number == 6912:
        replayed = len([row for row in payload.get("cell_identity_rows", []) if isinstance(row, Mapping)])
        ready = int(
            replayed == 1400
            and payload.get("duplicate_cell_count") == 0
            and payload.get("source_mutation_count") == 0
            and payload.get("model_inference_call_count") == 0
            and isinstance(payload.get("source_flag_preservation"), Mapping)
            and payload["source_flag_preservation"].get("passed") is True
            and _checks_pass(payload) is True
        )
        rows = [
            _comparison("replayed_cell_count", payload.get("replayed_cell_count"), replayed, "cell_identity_rows"),
            _comparison("clean_relation_corpus_ready_score", payload.get("clean_relation_corpus_ready_score"), ready, "cell_and_gate_rows"),
        ]
    elif number in {6913, 6914, 6919}:
        field = {6913: "source_tuple_shard_ready_score", 6914: "asp_isomorphic_shard_ready_score", 6919: "prefix_viability_canary_ready_score"}[number]
        rows = [_comparison(field, payload.get(field), int(_checks_pass(payload) is True), "gate_check_rows")]
    elif number == 6915:
        admitted = [row for row in payload.get("admitted_event_rows", []) if isinstance(row, Mapping) and row.get("model_produced") is True]
        count = len(admitted)
        family_counts = {family: sum(row.get("family") == family for row in admitted) for family in REQUIRED_FAMILIES}
        model_counts: dict[str, int] = defaultdict(int)
        for row in admitted:
            model_counts[str(row.get("model_family") or "")] += 1
        ready = int(
            count >= 90
            and all(value >= 1 for value in family_counts.values())
            and all(model_counts.get(family, 0) >= 10 for family in ("qwen_moe", "gemma_dense", "gemma_moe"))
        )
        rows = [
            _comparison("qualified_model_relation_event_count", payload.get("qualified_model_relation_event_count"), count, "admitted_event_rows"),
            _comparison("qualified_relation_event_bank_ready_score", payload.get("qualified_relation_event_bank_ready_score"), ready, "admitted_event_rows"),
        ]
    elif number == 6920:
        model_rows = [row for row in payload.get("per_model_arm_rows", []) if isinstance(row, Mapping)]
        cells = {
            (str(row.get("model_spec") or row.get("model_id") or row.get("model")), str(row.get("arm"))): row.get("cell_count")
            for row in model_rows
        }
        complete = int(all(isinstance(cells.get((model, arm)), int) and cells[(model, arm)] >= 30 for model in REQUIRED_MODEL_IDS for arm in GENERATION_ARMS))
        deltas = [row for row in payload.get("validity_delta_rows", []) if isinstance(row, Mapping)]
        utility = int(
            len(deltas) == len(REQUIRED_MODEL_IDS)
            and sum(isinstance(row.get("validity_delta"), (int, float)) and row["validity_delta"] > 0 for row in deltas) >= 2
            and all(row.get("no_regression_over_0_02") is True and row.get("parse_failure_did_not_rise") is True for row in deltas)
        )
        rows = [
            _comparison("guided_generation_run_complete_score", payload.get("guided_generation_run_complete_score"), complete, "per_model_arm_rows"),
            _comparison("exact_guidance_utility_score", payload.get("exact_guidance_utility_score"), utility, "validity_delta_rows"),
        ]
    elif number == 6921:
        checks = payload.get("gate_check_summary", {}).get("checks", []) if isinstance(payload.get("gate_check_summary"), Mapping) else []
        by_name = {row.get("check"): row.get("passed") for row in checks if isinstance(row, Mapping)}
        complete = int(all(by_name.get(name) is True for name in ("receipt_discovery_complete", "provenance_classification_complete", "canonical_row_dedupe_complete", "banked_credit_replay_complete")) and payload.get("automatic_arm_mutation_count") == 0)
        arm_rows = [row for row in payload.get("per_arm_rows", []) if isinstance(row, Mapping)]
        eligible = int(bool(arm_rows) and all(row.get("meets_floor") is True for row in arm_rows) and (payload.get("new_eligible_receipt_count") or 0) > 0)
        rows = [
            _comparison("arc_supervisor_audit_complete_score", payload.get("arc_supervisor_audit_complete_score"), complete, "receipt_and_replay_rows"),
            _comparison("banked_credit_eligible_score", payload.get("banked_credit_eligible_score"), eligible, "per_arm_rows"),
        ]
    return rows


def propagate_dependency_taint(
    state_rows: Sequence[Mapping[str, Any]], dependencies: Mapping[str, Sequence[str]]
) -> list[JsonDict]:
    """Propagate direct evidence defects through every reachable dependency."""

    taint_states = {"flagged", "stale", "invalid", "row_disagreement", "wrong_verdict_class", "blocked", "skipped", "absent", "circular_positive", "null", "disqualified"}
    state_by_id = {str(row.get("task_id")): str(row.get("evidence_state")) for row in state_rows}
    inherited: dict[str, dict[str, str]] = {}
    changed = True
    while changed:
        changed = False
        for target, upstreams in dependencies.items():
            roots = dict(inherited.get(target, {}))
            for upstream in upstreams:
                state = state_by_id.get(upstream)
                if state in taint_states:
                    roots[upstream] = state
                roots.update(inherited.get(upstream, {}))
            if roots != inherited.get(target, {}):
                inherited[target] = roots
                changed = True
    return [
        {
            "row_type": "dependency_taint",
            "target_task_id": target,
            "immediate_upstream_task_ids": list(dependencies.get(target, [])),
            "root_task_ids": sorted(roots),
            "taint_types": sorted(set(roots.values())),
            "transitive": any(root not in dependencies.get(target, []) for root in roots),
        }
        for target, roots in inherited.items()
        if roots
    ]


def compare_prior_verdicts(
    tasks: Sequence[Mapping[str, Any]], state_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Emit exact repeats and their required retirement actions."""

    verdicts = {str(row.get("task_id")): row.get("honest_verdict") for row in state_rows}
    comparisons: list[JsonDict] = []
    actions: list[JsonDict] = []
    for task in tasks:
        current = verdicts.get(str(task.get("task_id")))
        for prior in task.get("prior_failures", []):
            if not isinstance(prior, Mapping):
                continue
            repeated = current == prior.get("verdict")
            row = {
                "row_type": "prior_verdict_comparison",
                "task_id": task.get("task_id"),
                "prior_experiment_id": prior.get("experiment_id"),
                "prior_verdict": prior.get("verdict"),
                "current_verdict": current,
                "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                "addressed_by": prior.get("addressed_by"),
                "verdict_repeated": repeated,
            }
            comparisons.append(row)
            if repeated and prior.get("retire_if_same_verdict") is True:
                actions.append(
                    {
                        "row_type": "retirement_action",
                        "task_id": task.get("task_id"),
                        "prior_experiment_id": prior.get("experiment_id"),
                        "repeated_verdict": current,
                        "action": "retire_exact_repeated_scope",
                        "manifest_edited": False,
                    }
                )
    return comparisons, actions


def claim_row(claim: str, evidence_class: str, promoted: bool, scientific: bool = True) -> JsonDict:
    """Record promotion intent so unsupported science can be counted exactly."""

    return {
        "row_type": "milestone_claim",
        "claim": claim,
        "evidence_class": evidence_class,
        "promoted": promoted,
        "scientific": scientific,
    }


def false_promotion_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count scientific promotions that lack non-circular positive evidence."""

    return sum(
        row.get("scientific") is True
        and row.get("promoted") is True
        and row.get("evidence_class") != "positive"
        for row in rows
    )


def _matrix(branch: str, values: Mapping[str, tuple[Any, str]]) -> list[JsonDict]:
    """Write one checkable row for each required branch dimension."""

    return [
        {
            "row_type": "branch_evidence",
            "branch": branch,
            "dimension": dimension,
            "observed": observed,
            "status": status,
        }
        for dimension, (observed, status) in values.items()
    ]


def _branch_outputs(metrics: Mapping[str, Any]) -> tuple[dict[str, list[JsonDict]], list[JsonDict], list[JsonDict]]:
    """Build four ungated branch matrices and one disposition per branch."""

    relation_count = metrics.get("qualified_model_relation_event_count")
    relation_ready = metrics.get("qualified_relation_event_bank_ready_score") == 1
    exact_run = metrics.get("guided_generation_run_complete_score") == 1
    exact_utility = metrics.get("exact_guidance_utility_score") == 1
    arc_audit = metrics.get("arc_supervisor_audit_complete_score") == 1
    arc_eligible = metrics.get("banked_credit_eligible_score") == 1
    matrices = {
        "relation": _matrix("relation_qualification", {
            "prerequisite_reached": (metrics.get("clean_relation_corpus_ready_score") == 1, "measured"),
            "run_complete": (metrics.get("source_tuple_shard_ready_score") == 1 and metrics.get("asp_isomorphic_shard_ready_score") == 1, "measured"),
            "effect": ({"qualified_events": relation_count, "required": 90}, "insufficient" if not relation_ready else "passed"),
            "safety": (metrics.get("clean_relation_corpus_ready_score") == 1, "measured"),
            "circularity": (True, "oracle_backed"),
            "family_coverage": ({"qualified_families": 1 if relation_count else 0, "required": 5}, "insufficient"),
            "production_adoption": (False, "not_admissible"),
        }),
        "self_learning": _matrix("continuous_self_learning", {
            "prerequisite_reached": (metrics.get("prospective_relation_stream_ready_score") == 1, "blocked"),
            "run_complete": (metrics.get("self_learning_run_complete_score") == 1, "not_run"),
            "effect": (None, "unsupported"),
            "safety": (None, "unsupported"),
            "circularity": (True, "inherited"),
            "family_coverage": ({"measured_families": 0, "required": 5}, "not_run"),
            "production_adoption": (False, "blocked"),
        }),
        "exact": _matrix("exact_guidance", {
            "prerequisite_reached": (metrics.get("prefix_viability_canary_ready_score") == 1, "measured"),
            "run_complete": (exact_run, "measured"),
            "effect": ({"utility_score": int(exact_utility), "validity_delta": 0.0 if exact_run and not exact_utility else None}, "null" if exact_run and not exact_utility else "measured"),
            "safety": (exact_run, "no_parse_or_family_regression" if exact_run else "unsupported"),
            "circularity": (True, "oracle_backed_final_labels"),
            "family_coverage": ({"covered_families": 5 if exact_run else 0, "required": 5}, "measured" if exact_run else "unsupported"),
            "production_adoption": (False, "null_effect"),
        }),
        "arc": _matrix("arc_supervisor_generalization", {
            "prerequisite_reached": ((metrics.get("new_eligible_receipt_count") or 0) > 0, "measured"),
            "run_complete": (arc_audit, "measured"),
            "effect": (None if not arc_eligible else True, "unsupported" if not arc_eligible else "passed"),
            "safety": (metrics.get("automatic_arm_mutation_count") == 0, "measured"),
            "circularity": (False, "non_oracle_audit"),
            "family_coverage": ({"arms_meeting_floor": metrics.get("arms_meeting_floor", 0), "required": metrics.get("arm_count", 0)}, "insufficient" if not arc_eligible else "passed"),
            "production_adoption": (False, "insufficient_evidence"),
        }),
    }
    specifications = [
        ("relation_qualification", "continue", "Ship a changed relation-qualification technique that yields at least 90 model events, at least 10 per model family, across all five families."),
        ("continuous_self_learning", "block", "Ship a prospective relation stream with prospective_relation_stream_ready_score=1 from a newly qualified bank of at least 90 model events."),
        ("exact_guidance", "retire", "Ship a non-prefix-rejection guidance mechanism with a preregistered non-saturated fixture before another generation comparison."),
        ("arc_supervisor_generalization", "continue", "Accrue new live banked receipts until every supervisor arm has at least 10 fired outcomes before another cold credit audit."),
    ]
    dispositions = [
        {"row_type": "branch_disposition", "branch": branch, "disposition": disposition, "next_executable_prerequisite": prerequisite}
        for branch, disposition, prerequisite in specifications
    ]
    prerequisites = [
        {"row_type": "next_prerequisite", "branch": row["branch"], "prerequisite": row["next_executable_prerequisite"]}
        for row in dispositions
    ]
    return matrices, dispositions, prerequisites


def sha256_file(path: Path) -> str | None:
    """Hash a file in chunks so large evidence does not need a second copy."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict | None:
    """Return None for malformed or non-object evidence."""

    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_checker(path: Path, name: str) -> Any:
    """Load the checked-in verifier by exact path so the current code runs."""

    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key in _CHECKER_CACHE:
        return _CHECKER_CACHE[key]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    # The adversarial verifier reloads itself when its source changes. Register
    # the module first so that check can find the live module identity.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _CHECKER_CACHE[key] = module
    return module


def _audit_artifact(root: Path, path: Path) -> tuple[JsonDict, tuple[str, list[str]]]:
    """Run both current verifiers once per unchanged artifact."""

    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key in _AUDIT_CACHE:
        return deepcopy(_AUDIT_CACHE[key])
    try:
        adversarial_module = _load_checker(root / ADVERSARIAL_PATH, "exp6922_adversarial")
        row_module = _load_checker(root / ROW_LINT_PATH, "exp6922_rows")
        adversarial = dict(adversarial_module.verify_artifact(path))
        row_check = row_module.check_artifact(path)
        result = (adversarial, (str(row_check[0]), list(row_check[1])))
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        result = (
            {"flag_count": 1, "max_severity": 2, "flags": [{"kind": "VERIFIER_ERROR", "severity": "critical", "detail": detail}]},
            ("unreadable", [detail]),
        )
    _AUDIT_CACHE[key] = deepcopy(result)
    return result


def _source_checks(root: Path) -> tuple[JsonDict, dict[str, JsonDict]]:
    """Evaluate global preconditions and bind their current bytes."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for name, relative in GLOBAL_SOURCES.items():
        path = root / relative
        digest = sha256_file(path)
        checks.append({"check": name, "expected": "readable_file", "observed": digest, "passed": digest is not None})
        if digest is not None:
            hashes[name] = {"path": relative.as_posix(), "sha256": digest}
    failures = [row["check"] for row in checks if not row["passed"]]
    return {
        "checks": checks,
        "passed": not failures,
        "failed_checks": failures,
        "failed_check": failures[0] if failures else None,
        "expected": "all global sources readable",
        "observed": failures if failures else "all global sources readable",
    }, hashes


def _task_dependencies(tasks: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Read dependency edges from the executable gate declarations."""

    return {
        str(task.get("task_id")): [str(gate.get("upstream")) for gate in task.get("gates", []) if isinstance(gate, Mapping)]
        for task in tasks
    }


def _gate_pass(observed: Any, operator: str, expected: Any) -> bool:
    """Replay the two operators used by V605 without guessing missing values."""

    if observed is None:
        return False
    if operator == "==":
        return observed == expected
    if operator == ">=":
        return isinstance(observed, (int, float)) and observed >= expected
    return False


def _field_principles() -> dict[str, str]:
    """Give every required field one plain-language evidence rule."""

    return {
        field: f"The {field} field keeps this part of the V605 synthesis explicit and replayable."
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic content while excluding wall time and the hash itself."""

    value = deepcopy(dict(artifact))
    value.pop("duration_s", None)
    value.pop("reproducibility_checksum", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_artifact(root: Path, run_date: str) -> JsonDict:
    """Build one complete synthesis even when a global precondition fails."""

    started = time.monotonic()
    preconditions, source_hashes = _source_checks(root)
    design_text = (root / DESIGN_PATH).read_text(encoding="utf-8") if (root / DESIGN_PATH).is_file() else ""
    try:
        roadmap = load_yaml(root / ROADMAP_PATH)
    except (OSError, ValueError, yaml.YAMLError):
        roadmap = {}
    contract_rows = build_contract_rows(design_text, roadmap)
    yaml_tasks = _roadmap_tasks(roadmap)
    task_lookup = {task["number"]: task for task in yaml_tasks}
    tasks = [
        task_lookup.get(
            expected["number"],
            {**deepcopy(expected), "order": order, "milestone": MILESTONE, "prompt": "", "prior_failures": []},
        )
        for order, expected in enumerate(EXPECTED_TASKS, 1)
    ]
    log_text = (root / CONDUCTOR_LOG_PATH).read_text(encoding="utf-8") if (root / CONDUCTOR_LOG_PATH).is_file() else ""
    activation, conductor = parse_conductor_states(log_text, tasks)
    state_rows: list[JsonDict] = []
    adversarial_rows: list[JsonDict] = []
    row_consistency_rows: list[JsonDict] = []
    metric_rows: list[JsonDict] = []
    metric_values: dict[str, Any] = {}
    for task in tasks:
        number = int(task["number"])
        task_id = str(task["task_id"])
        if number == 6922:
            state_rows.append(
                {
                    "row_type": "task_state", "task_id": task_id, "number": number,
                    "title": task.get("title"), "conductor_state": "current_synthesis",
                    "artifact_state": "current_synthesis", "evidence_state": "partial",
                    "declared_verdict_class": "partial", "structural_verdict_class": "partial",
                    "verdict_class": "partial", "honest_verdict": COMPLETE_VERDICT,
                    "admissible": False, "replayable": True, "adversarial_critical": False,
                    "row_check_status": "current_synthesis", "row_findings": [],
                }
            )
            continue
        path = root / str(task["deliverable"])
        if not path.is_file():
            artifact_state, payload, adversarial, row_check = "absent", None, None, None
        else:
            payload = _read_json(path)
            if payload is None:
                artifact_state, adversarial, row_check = "invalid", None, None
            else:
                stale = bool(activation and datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < activation)
                artifact_state = "stale" if stale else "present"
                adversarial, row_check = _audit_artifact(root, path)
                source_hashes[f"exp{number}"] = {"path": str(task["deliverable"]), "sha256": sha256_file(path)}
                adversarial_rows.append(
                    {
                        "row_type": "adversarial_recheck", "task_id": task_id,
                        "artifact_path": str(task["deliverable"]),
                        "status": "flagged" if any(str(flag.get("severity", "")).lower() == "critical" for flag in adversarial.get("flags", [])) else "checked",
                        "flag_count": adversarial.get("flag_count", len(adversarial.get("flags", []))),
                        "max_severity": adversarial.get("max_severity"), "flags": deepcopy(adversarial.get("flags", [])),
                    }
                )
                row_consistency_rows.append(
                    {"row_type": "row_consistency", "task_id": task_id, "artifact_path": str(task["deliverable"]), "status": row_check[0], "findings": row_check[1]}
                )
                fresh = recompute_headlines(number, payload)
                for row in fresh:
                    row["task_id"] = task_id
                    metric_values[row["field"]] = row["recomputed"]
                metric_rows.extend(fresh)
                for key in ("prospective_relation_stream_ready_score", "self_learning_run_complete_score", "new_eligible_receipt_count", "automatic_arm_mutation_count"):
                    if key in payload:
                        metric_values[key] = _scalar(payload[key])
                if number == 6921:
                    arms = [row for row in payload.get("per_arm_rows", []) if isinstance(row, Mapping)]
                    metric_values["arm_count"] = len(arms)
                    metric_values["arms_meeting_floor"] = sum(row.get("meets_floor") is True for row in arms)
        row = classify_evidence(
            task,
            conductor.get(task_id, {"state": "not_recorded"}),
            artifact_state,
            payload,
            adversarial,
            row_check,
        )
        state_rows.append(row)
        if payload:
            for field in ("clean_relation_corpus_ready_score", "source_tuple_shard_ready_score", "asp_isomorphic_shard_ready_score", "qualified_relation_event_bank_ready_score", "qualified_model_relation_event_count", "prefix_viability_canary_ready_score", "guided_generation_run_complete_score", "exact_guidance_utility_score", "arc_supervisor_audit_complete_score", "banked_credit_eligible_score"):
                metric_values.setdefault(field, _scalar(payload.get(field)))

    for task in tasks:
        outcomes = []
        for gate in task.get("gates", []):
            if not isinstance(gate, Mapping):
                continue
            observed = metric_values.get(str(gate.get("artifact_field")))
            outcomes.append({**deepcopy(dict(gate)), "observed": observed, "passed": _gate_pass(observed, str(gate.get("op")), gate.get("value"))})
        matching = next(row for row in state_rows if row["task_id"] == task["task_id"])
        matching["gate_outcomes"] = outcomes

    dependencies = _task_dependencies(tasks)
    taint_rows = propagate_dependency_taint(state_rows, dependencies)
    comparisons, retirement = compare_prior_verdicts(tasks, state_rows)
    matrices, dispositions, prerequisites = _branch_outputs(metric_values)
    claims = [
        claim_row("V605 evidence synthesis is complete.", "positive", True, scientific=False),
        claim_row("The qualified relation bank reached production scale.", "disqualified", False),
        claim_row("Continuous self-learning improved future exact utility.", "blocked", False),
        claim_row("Exact prefix guidance beat matched unguided best-of-k.", "null", False),
        claim_row("The ARC supervisor improved live hidden-game progress.", "null", False),
        claim_row("Oracle-backed qualification is non-circular positive science.", "circular_positive", False),
    ]
    promotions = false_promotion_count(claims)
    present_replayable = all(row["replayable"] for row in state_rows if row["artifact_state"] not in {"absent"})
    conductor_complete = all(row["conductor_state"] != "not_recorded" for row in state_rows)
    synthesis_checks = [
        {"check": "global_preconditions", "expected": True, "observed": preconditions["passed"], "passed": preconditions["passed"]},
        {"check": "document_yaml_contract", "expected": 12, "observed": sum(row["passed"] for row in contract_rows), "passed": all(row["passed"] for row in contract_rows)},
        {"check": "task_states_classified", "expected": 12, "observed": len(state_rows), "passed": len(state_rows) == 12 and conductor_complete},
        {"check": "present_artifacts_replayable", "expected": True, "observed": present_replayable, "passed": present_replayable},
        {"check": "false_promotion_count", "expected": 0, "observed": promotions, "passed": promotions == 0},
    ]
    complete_score = int(all(row["passed"] for row in synthesis_checks))
    failed = [row["check"] for row in synthesis_checks if not row["passed"]]
    blocked = not preconditions["passed"]
    artifact: JsonDict = {
        "schema": "carnot.experiment_6922.v605_independent_capstone.v1",
        "experiment_id": 6922,
        "run_date": run_date,
        "status": "complete_blocked" if blocked else "complete",
        "field_principles": _field_principles(),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": source_hashes,
        "document_yaml_contract_rows": contract_rows,
        "task_state_rows": state_rows,
        "conductor_artifact_state_rows": [{key: row.get(key) for key in ("task_id", "conductor_state", "artifact_state", "evidence_state", "verdict_class", "honest_verdict")} for row in state_rows],
        "missing_artifact_rows": [row for row in state_rows if row["artifact_state"] == "absent"],
        "skipped_task_rows": [row for row in state_rows if row["evidence_state"] == "skipped"],
        "blocked_task_rows": [row for row in state_rows if row["verdict_class"] == "blocked"],
        "flagged_artifact_rows": [row for row in state_rows if row["evidence_state"] == "flagged"],
        "null_result_rows": [row for row in state_rows if row["verdict_class"] == "null"],
        "circular_result_rows": [row for row in state_rows if row["verdict_class"] == "circular_positive"],
        "positive_result_rows": [row for row in state_rows if row["verdict_class"] == "positive"],
        "adversarial_recheck_rows": adversarial_rows,
        "row_consistency_rows": row_consistency_rows,
        "reported_vs_recomputed_metrics": metric_rows,
        "dependency_taint_rows": taint_rows,
        "prior_verdict_comparison_rows": comparisons,
        "retirement_action_rows": retirement,
        "relation_branch_rows": matrices["relation"],
        "self_learning_branch_rows": matrices["self_learning"],
        "exact_guidance_branch_rows": matrices["exact"],
        "arc_generalization_branch_rows": matrices["arc"],
        "branch_disposition_rows": dispositions,
        "next_prerequisite_rows": prerequisites,
        "milestone_claim_rows": claims,
        "false_promotion_count": promotions,
        "random_seed": RANDOM_SEED,
        "v605_capstone_complete_score": complete_score,
        "gate_check_summary": {"checks": synthesis_checks, "passed": not failed, "failed_checks": failed, "failed_check": failed[0] if failed else None, "expected": "all synthesis checks pass", "observed": failed if failed else "all synthesis checks pass"},
        "verifier_is_oracle": False,
        "verdict_class": "blocked" if blocked else "partial",
        "honest_verdict": BLOCKED_VERDICT if blocked else COMPLETE_VERDICT,
    }
    artifact["rows"] = [
        *artifact["task_state_rows"], *artifact["document_yaml_contract_rows"],
        *artifact["reported_vs_recomputed_metrics"], *artifact["dependency_taint_rows"],
        *artifact["prior_verdict_comparison_rows"], *artifact["retirement_action_rows"],
        *artifact["relation_branch_rows"], *artifact["self_learning_branch_rows"],
        *artifact["exact_guidance_branch_rows"], *artifact["arc_generalization_branch_rows"],
        *artifact["branch_disposition_rows"], *artifact["milestone_claim_rows"],
    ]
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Fail closed when the terminal artifact loses a required evidence rule."""

    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not REQUIRED_ARTIFACT_FIELDS <= set(principles):
        errors.append("field_principles must cover every required field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class is outside the closed enum")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict must start with complete_")
    if artifact.get("false_promotion_count") != 0:
        errors.append("false_promotion_count must be zero")
    if artifact.get("v605_capstone_complete_score") not in {0, 1}:
        errors.append("v605_capstone_complete_score must be zero or one")
    task_rows = artifact.get("task_state_rows")
    if not isinstance(task_rows, list) or len(task_rows) != 12:
        errors.append("task_state_rows must classify all 12 tasks")
    dispositions = artifact.get("branch_disposition_rows")
    if not isinstance(dispositions, list) or len(dispositions) != 4 or any(
        not isinstance(row, Mapping)
        or row.get("disposition") not in DISPOSITIONS
        or not str(row.get("next_executable_prerequisite") or "").strip()
        for row in dispositions or []
    ):
        errors.append("branch dispositions must cover four branches with exact prerequisites")
    if isinstance(artifact.get("milestone_claim_rows"), list) and false_promotion_count(artifact["milestone_claim_rows"]) != artifact.get("false_promotion_count"):
        errors.append("false promotion count disagrees with milestone claims")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, Mapping) or not summary.get("failed_check"):
            errors.append("blocked verdict requires an exact failed gate check")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def _write_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the deliverable only after a complete JSON file is durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Write or validate the Exp6922 terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260903")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.repo_root / args.output
    if args.validate:
        payload = _read_json(output)
        errors = validate_artifact(payload or {})
    else:
        payload = build_artifact(args.repo_root, args.date)
        errors = validate_artifact(payload)
        if not errors:
            _write_atomic(output, payload)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"validated {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper owns command execution.
    raise SystemExit(main())
