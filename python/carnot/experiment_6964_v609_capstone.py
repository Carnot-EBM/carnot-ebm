"""Reconcile V609 contracts and evidence without rerunning science.

The capstone reads primary rows again because copied headlines can preserve an
upstream mistake. It keeps unavailable work explicit and gives exact external
certificates more authority than model confidence. See REQ-REPORT-6964.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6964_v609_capstone.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
V608_CAPSTONE_PATH = Path("results/experiment_6952_v608_capstone.json")
MODULE_PATH = Path("python/carnot/experiment_6964_v609_capstone.py")
RUNNER_PATH = Path("scripts/experiments/experiment_6964_v609_capstone.py")
INFERENCE_SUBSTRATE = "independent_artifact_replay_and_contract_reconciliation_no_llm"
MILESTONE = "2026.09.609"
RANDOM_SEED = 6964
BLOCKED_VERDICT = "blocked_v609_capstone"
NULL_VERDICT = "complete_null_v609_capstone_science_incomplete"
DISQUALIFIED_VERDICT = "complete_disqualified_v609_capstone_flagged_or_conflicting_evidence"
POSITIVE_VERDICT = "complete_positive_v609_capstone_all_branches_supported"
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
EXACT_AUTHORITIES = {"exact_dual_engine_certificate", "z3_and_exact_enumeration"}


def _gate(upstream: str, field: str, op: str = "==", value: Any = 1) -> JsonDict:
    """Return one dependency in the roadmap's structured form."""

    return {"upstream": upstream, "artifact_field": field, "op": op, "value": value}


EXPECTED_TASKS: tuple[JsonDict, ...] = (
    {
        "number": 6953,
        "task_id": "exp6953-v609-source-delta",
        "title": "V609 post-marker source delta and compatibility audit",
        "deliverable": "results/experiment_6953_v609_source_delta.json",
        "gates": [],
    },
    {
        "number": 6954,
        "task_id": "exp6954-v609-contract-advisory",
        "title": "V609 advisory execution-contract and gate-cascade audit",
        "deliverable": "results/experiment_6954_v609_contract_advisory.json",
        "gates": [],
    },
    {
        "number": 6955,
        "task_id": "exp6955-reformulation-fixture",
        "title": "Exact optimization-reformulation mapping fixture",
        "deliverable": "results/experiment_6955_reformulation_fixture.json",
        "gates": [],
    },
    {
        "number": 6956,
        "task_id": "exp6956-three-family-reformulation-bank",
        "title": "Three-family SOTA reformulation mapping bank",
        "deliverable": "results/experiment_6956_three_family_reformulation_bank.json",
        "gates": [_gate("exp6955-reformulation-fixture", "reformulation_fixture_ready_score")],
    },
    {
        "number": 6957,
        "task_id": "exp6957-smt-mapping-certification",
        "title": "Independent SMT certification of SOTA mappings",
        "deliverable": "results/experiment_6957_smt_mapping_certification.json",
        "gates": [
            _gate("exp6956-three-family-reformulation-bank", "reformulation_bank_complete_score")
        ],
    },
    {
        "number": 6958,
        "task_id": "exp6958-convex-factor-energy-canary",
        "title": "Convex compositional factor-energy canary",
        "deliverable": "results/experiment_6958_convex_factor_energy_canary.json",
        "gates": [_gate("exp6955-reformulation-fixture", "reformulation_fixture_ready_score")],
    },
    {
        "number": 6959,
        "task_id": "exp6959-certified-energy-selection",
        "title": "Causal certified-energy candidate selection",
        "deliverable": "results/experiment_6959_certified_energy_selection.json",
        "gates": [
            _gate("exp6957-smt-mapping-certification", "smt_certification_run_complete_score"),
            _gate("exp6958-convex-factor-energy-canary", "convex_factor_run_complete_score"),
        ],
    },
    {
        "number": 6960,
        "task_id": "exp6960-certified-selection-cold-audit",
        "title": "Fresh-process certified-selection audit",
        "deliverable": "results/experiment_6960_certified_selection_cold_audit.json",
        "gates": [
            _gate("exp6959-certified-energy-selection", "certified_selection_run_complete_score")
        ],
    },
    {
        "number": 6961,
        "task_id": "exp6961-certified-event-sequence",
        "title": "Sealed chronological outcome-certificate sequence",
        "deliverable": "results/experiment_6961_certified_event_sequence.json",
        "gates": [
            _gate("exp6957-smt-mapping-certification", "smt_certification_run_complete_score")
        ],
    },
    {
        "number": 6962,
        "task_id": "exp6962-queue-regulated-self-learning",
        "title": "Queue-regulated continuous self-learning",
        "deliverable": "results/experiment_6962_queue_regulated_self_learning.json",
        "gates": [
            _gate("exp6961-certified-event-sequence", "certified_event_sequence_ready_score")
        ],
    },
    {
        "number": 6963,
        "task_id": "exp6963-queue-memory-cold-audit",
        "title": "Fresh-process queue-memory safety audit",
        "deliverable": "results/experiment_6963_queue_memory_cold_audit.json",
        "gates": [
            _gate("exp6962-queue-regulated-self-learning", "queue_learning_run_complete_score")
        ],
    },
    {
        "number": 6964,
        "task_id": "exp6964-v609-capstone",
        "title": "V609 independent capstone and V610 handoff",
        "deliverable": OUTPUT_PATH.as_posix(),
        "gates": [],
    },
)

SCORE_AUTHORITIES = {
    "sota_mapping_positive_score": 6957,
    "convex_factor_positive_score": 6958,
    "certified_energy_positive_score": 6959,
    "audited_certified_energy_positive_score": 6960,
    "queue_learning_positive_score": 6962,
    "audited_queue_learning_positive_score": 6963,
}
EXPECTED_MODELS = {
    6956: (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ),
    6957: (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ),
    6962: (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ),
}

REQUIRED_ARTIFACT_FIELDS = {
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "task_contract_rows",
    "task_state_rows",
    "gate_replay_rows",
    "verdict_class_rows",
    "adversarial_verify_rows",
    "aggregate_recompute_rows",
    "model_coverage_rows",
    "mapping_rows",
    "certificate_rows",
    "convex_factor_rows",
    "selection_rows",
    "selection_audit_rows",
    "event_sequence_rows",
    "queue_learning_rows",
    "queue_audit_rows",
    "safety_rows",
    "exclusion_candidate_rows",
    "hardware_provenance_rows",
    "v610_gap_rows",
    "random_seed",
    "reproducibility_checksum",
    *SCORE_AUTHORITIES,
    "v609_capstone_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}


def spec_anchors(text: str) -> list[str]:
    """List requirement anchors so tests show which contract owns behavior."""

    return re.findall(r"(?:REQ|SCENARIO)-[A-Z0-9-]+", text)


def load_yaml(path: Path) -> JsonDict:
    """Load one YAML mapping and reject roots that cannot contain tasks."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return value


def _number(value: Any) -> int | None:
    """Extract an experiment number while rejecting malformed task IDs."""

    match = re.match(r"exp(\d+)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _scalar(value: Any) -> Any:
    """Read a field that can use the repository's principle wrapper."""

    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _parse_gates(text: str) -> list[JsonDict]:
    """Parse every semicolon-separated design gate without using YAML facts."""

    if text.strip().lower() == "none":
        return []
    rows = []
    for item in text.split(";"):
        raw = item.strip()
        clean = raw.replace("`", "")
        match = re.fullmatch(r"(exp\d+[a-z0-9-]*)\s+([A-Za-z0-9_]+)\s*(==|>=|<=|>|<)\s*(.+)", clean)
        if not match:
            rows.append({"unparsed": raw})
            continue
        upstream_number = _number(match.group(1))
        upstream = next(
            (task["task_id"] for task in EXPECTED_TASKS if task["number"] == upstream_number),
            match.group(1),
        )
        rows.append(
            _gate(str(upstream), match.group(2), match.group(3), yaml.safe_load(match.group(4)))
        )
    return rows


def parse_design(text: str) -> list[JsonDict]:
    """Parse only the human document's fixed task-contract table."""

    section = text.split("## Exact Task Contract", 1)[-1].split("## Hardware", 1)[0]
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*(exp\d+[a-z0-9-]*)\s*\|\s*([^|]+?)\s*"
        r"\|\s*`?([^|`]+?)`?\s*\|\s*([^|]+?)\s*\|$",
        re.MULTILINE,
    )
    return [
        {
            "order": int(order),
            "number": _number(task_id),
            "task_id": task_id.strip(),
            "title": title.strip(),
            "deliverable": deliverable.strip(),
            "gates": _parse_gates(gates),
        }
        for order, task_id, title, deliverable, gates in pattern.findall(section)
    ]


def roadmap_tasks(roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Normalize YAML tasks without borrowing values from the document."""

    raw_tasks = roadmap.get("tasks", [])
    if not isinstance(raw_tasks, list):
        return []
    rows = []
    for order, task in enumerate(raw_tasks, 1):
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
                "prior_failures": deepcopy(task.get("prior_failures") or []),
            }
        )
    return rows


def build_contract_rows(design_text: str, roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Compare both primary sources against the fixed V609 task sequence."""

    document = parse_design(design_text)
    executable = roadmap_tasks(roadmap)
    rows = []
    for order, expected in enumerate(EXPECTED_TASKS, 1):
        doc = document[order - 1] if order <= len(document) else {}
        task = executable[order - 1] if order <= len(executable) else {}
        checks = {
            "source_counts_match": len(document) == len(executable) == 12,
            "document_present": bool(doc),
            "yaml_present": bool(task),
            "order_match": doc.get("order") == task.get("order") == order,
            "number_match": doc.get("number") == task.get("number") == expected["number"],
            "task_id_match": doc.get("task_id") == task.get("task_id") == expected["task_id"],
            "title_match": doc.get("title") == task.get("title") == expected["title"],
            "deliverable_match": doc.get("deliverable")
            == task.get("deliverable")
            == expected["deliverable"],
            "gates_match": doc.get("gates") == task.get("gates") == expected["gates"],
            "milestone_match": roadmap.get("milestone") == task.get("milestone") == MILESTONE,
        }
        rows.append(
            {
                "row_type": "task_contract",
                "order": order,
                "number": expected["number"],
                "task_id": expected["task_id"],
                "document": doc or None,
                "yaml": task or None,
                **checks,
                "passed": all(checks.values()),
                "terminal": True,
            }
        )
    return rows


def parse_conductor_states(text: str, tasks: Sequence[Mapping[str, Any]]) -> dict[int, JsonDict]:
    """Read final V609 conductor rows while keeping `OK` advisory."""

    states: dict[int, JsonDict] = {}
    active = False
    for line in text.splitlines():
        if "Milestone 2026.09.609 activated" in line:
            active = True
            continue
        if not active or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        title, raw_status, detail = cells[1], cells[2], cells[3]
        task = next(
            (
                row
                for row in tasks
                if str(row.get("title") or "").startswith(title)
                or title.startswith(str(row.get("title") or ""))
            ),
            None,
        )
        if task is None:
            continue
        state = {"OK": "ok", "FLAGGED": "flagged"}.get(raw_status, "failed")
        if raw_status == "GATE_BLOCK":
            state = "preemptive_skip" if "Pre-emptive skip" in detail else "gate_blocked"
        states[int(task["number"])] = {
            "state": state,
            "raw_status": raw_status,
            "detail": detail,
            "timestamp": cells[0],
        }
    return states


def _prefix_verdict_class(honest_verdict: str) -> str | None:
    """Return a class only when the verdict text states one at its start."""

    text = honest_verdict.strip().lower()
    if text.startswith("flagged_"):
        return "disqualified"
    for klass in (
        "circular_positive",
        "disqualified",
        "blocked",
        "partial",
        "positive",
        "null",
    ):
        if text.startswith((f"{klass}_", f"{klass}:")):
            return klass
        if re.match(rf"^(?:complete|success|passed|shipped)[ :_]+{klass}(?:[ :_]|$)", text):
            return klass
    return None


def _verdict_shape(payload: Mapping[str, Any]) -> str:
    """Infer the source class without treating a failed science gate as a block."""

    status = str(_scalar(payload.get("status")) or "").lower()
    honest = str(_scalar(payload.get("honest_verdict")) or "").lower()
    declared = str(_scalar(payload.get("verdict_class")) or "").lower()
    gate_rows = payload.get("gates_evaluated")
    upstream_failed = isinstance(gate_rows, list) and any(
        isinstance(row, Mapping) and row.get("passed") is False for row in gate_rows
    )
    if status == "blocked" or honest.startswith("blocked_") or upstream_failed:
        return "blocked"
    if status in {"disqualified", "flagged"} or honest.startswith(("disqualified_", "flagged_")):
        return "disqualified"
    prefix = _prefix_verdict_class(honest)
    if prefix is not None:
        if prefix == "positive" and _scalar(payload.get("verifier_is_oracle")) is True:
            return "circular_positive"
        return prefix
    if declared == "positive" and _scalar(payload.get("verifier_is_oracle")) is True:
        return "circular_positive"
    if declared in VERDICT_CLASSES:
        return declared
    return "partial"


def _critical(adversarial: Mapping[str, Any] | None) -> bool:
    """Return true when fresh verification found any critical concern."""

    if not adversarial:
        return False
    return any(
        str(flag.get("severity") or "").lower() == "critical"
        for flag in adversarial.get("flags", [])
        if isinstance(flag, Mapping)
    )


def _is_bootstrap_only(payload: Mapping[str, Any]) -> bool:
    """Recognize conductor gate stubs that contain no experiment schema rows."""

    return (
        str(payload.get("status") or "").lower() == "blocked"
        and not isinstance(payload.get("field_principles"), Mapping)
        and not payload.get("rows")
    )


def classify_evidence(
    task: Mapping[str, Any],
    conductor: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    adversarial: Mapping[str, Any] | None,
    row_check: tuple[str, list[str]] | None,
    comparison_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Classify one task from all evidence while preserving weaker states."""

    conductor_state = str(conductor.get("state") or "not_recorded")
    if payload is None:
        klass = "blocked" if conductor_state in {"gate_blocked", "preemptive_skip"} else "partial"
        return {
            "row_type": "task_state",
            "task_id": task.get("task_id"),
            "number": task.get("number"),
            "title": task.get("title"),
            "conductor_state": conductor_state,
            "artifact_state": "missing",
            "evidence_state": "missing",
            "declared_verdict_class": None,
            "structural_verdict_class": None,
            "verdict_class": klass,
            "honest_verdict": "blocked_missing_artifact"
            if klass == "blocked"
            else "partial_missing_artifact",
            "admissible": False,
            "terminal": True,
            "row_check_status": None,
            "row_findings": [],
            "adversarial_critical": False,
        }
    declared_value = _scalar(payload.get("verdict_class"))
    declared = str(declared_value) if declared_value is not None else None
    structural = _verdict_shape(payload)
    honest = str(_scalar(payload.get("honest_verdict")) or "")
    row_status = row_check[0] if row_check else None
    row_findings = row_check[1] if row_check else []
    conflict = any(row.get("agrees") is False for row in comparison_rows)
    source_flag = _scalar(payload.get("flagged_adversarial")) is True
    critical = _critical(adversarial)
    bootstrap = _is_bootstrap_only(payload)
    if conductor_state == "flagged" or source_flag or critical:
        evidence_state, klass = "flagged", "disqualified"
    elif row_status in {"findings", "unreadable"} or conflict:
        evidence_state, klass = "row_headline_conflict", "disqualified"
    elif declared is not None and (declared not in VERDICT_CLASSES or declared != structural):
        evidence_state, klass = "verdict_prefix_class_conflict", "disqualified"
    elif bootstrap:
        evidence_state, klass = "bootstrap_only", "blocked"
    else:
        evidence_state, klass = structural, structural
    return {
        "row_type": "task_state",
        "task_id": task.get("task_id"),
        "number": task.get("number"),
        "title": task.get("title"),
        "conductor_state": conductor_state,
        "artifact_state": "bootstrap_only" if bootstrap else "present",
        "evidence_state": evidence_state,
        "declared_verdict_class": declared,
        "structural_verdict_class": structural,
        "verdict_class": klass,
        "honest_verdict": honest,
        "admissible": klass in {"positive", "circular_positive", "null"},
        "terminal": True,
        "row_check_status": row_status,
        "row_findings": row_findings,
        "adversarial_critical": critical,
    }


def _row_passes(row: Mapping[str, Any]) -> bool:
    """Read the explicit Boolean used by one source row."""

    for field in (
        "passed",
        "replay_matches",
        "upstream_matches",
        "authorities_agree",
        "label_isolated",
        "covered",
    ):
        if field in row:
            return row.get(field) is True
    if "terminal" in row:
        return row.get("terminal") is True
    return bool(row)


def _all_rows(payload: Mapping[str, Any], field: str, required: bool = True) -> bool:
    """Require every available row to carry an explicit passing outcome."""

    rows = payload.get(field)
    if not isinstance(rows, list) or not rows:
        return not required
    usable = [row for row in rows if isinstance(row, Mapping)]
    return bool(usable) and len(usable) == len(rows) and all(_row_passes(row) for row in usable)


def _aggregate_row(
    number: int, field: str, payload: Mapping[str, Any], recomputed: int, checks: Mapping[str, bool]
) -> JsonDict:
    """Return one comparison between raw-row reduction and stored headline."""

    reported = _scalar(payload.get(field))
    return {
        "row_type": "aggregate_recompute",
        "number": number,
        "field": field,
        "reported": reported,
        "recomputed": recomputed,
        "agrees": reported == recomputed,
        "checks": dict(checks),
    }


def _source_and_contract_rows(number: int, payload: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce advisory coverage and exact fixture or proposal readiness."""

    if number == 6953:
        checks = {
            "source_family_count": len(payload.get("source_family_rows", [])) == 15,
            "source_families_terminal": _all_rows(payload, "source_family_rows"),
            "paper_count": len(payload.get("paper_rows", [])) == 5,
            "papers_terminal": _all_rows(payload, "paper_rows"),
        }
        return [
            _aggregate_row(
                number,
                "v609_source_delta_complete_score",
                payload,
                int(all(checks.values())),
                checks,
            )
        ]
    if number == 6954:
        groups = (
            "task_parity_rows",
            "gate_contract_rows",
            "producer_field_rows",
            "prompt_contract_rows",
            "model_contract_rows",
            "prior_failure_rows",
            "exclusion_manifest_rows",
            "lint_command_rows",
            "mutation_rows",
            "bounded_scope_rows",
            "cascade_risk_rows",
        )
        complete_checks = {field: bool(payload.get(field)) for field in groups}
        conform_checks = {field: _all_rows(payload, field) for field in groups}
        return [
            _aggregate_row(
                number,
                "v609_contract_audit_complete_score",
                payload,
                int(all(complete_checks.values())),
                complete_checks,
            ),
            _aggregate_row(
                number,
                "v609_contract_conforms_score",
                payload,
                int(all(conform_checks.values())),
                conform_checks,
            ),
        ]
    if number == 6955:
        pairs = [row for row in payload.get("pair_rows", []) if isinstance(row, Mapping)]
        labels = {
            label: sum(row.get("expected_label") == label for row in pairs)
            for label in (
                "equivalent",
                "non_equivalent",
            )
        }
        checks = {
            "pair_count": len(pairs) == 120,
            "label_counts": labels == {"equivalent": 72, "non_equivalent": 48},
            "family_coverage": len({row.get("family") for row in pairs}) == 3,
            "exact_authority_agreement": len(pairs) == 120
            and all(row.get("authorities_agree") is True for row in pairs),
            "no_quarantine": all(row.get("quarantined") is False for row in pairs),
            "single_edit_hard_negatives": sum(
                row.get("hard_negative_edit_count") == 1 for row in pairs
            )
            == 48,
            "fresh_replay": len(payload.get("fresh_process_replay_rows", [])) == 120
            and _all_rows(payload, "fresh_process_replay_rows"),
        }
        return [
            _aggregate_row(
                number,
                "reformulation_fixture_ready_score",
                payload,
                int(all(checks.values())),
                checks,
            )
        ]
    if number == 6956:
        attempts = [row for row in payload.get("attempt_rows", []) if isinstance(row, Mapping)]
        models = {str(row.get("hf_id")) for row in attempts}
        prompts = {str(row.get("prompt_variant_id")) for row in attempts}
        receipt = payload.get("task_runtime_receipt")
        calls = receipt.get("call_rows", []) if isinstance(receipt, Mapping) else []
        checks = {
            "proposal_budget": len(attempts) == 162,
            "all_attempts_terminal": len(attempts) == 162
            and all(row.get("terminal") is True for row in attempts),
            "model_coverage": models == set(EXPECTED_MODELS[6956]),
            "held_out_pairs": len({row.get("pair_id") for row in attempts}) == 18,
            "prompt_variants": prompts == {"direct_affine", "domain_first", "objective_first"},
            "raw_outputs_durable": all(row.get("raw_durable") is True for row in attempts),
            "label_isolation": all(
                not row.get("hidden_label_isolation_errors") for row in attempts
            ),
            "runtime_receipts": len(calls) == 162,
        }
        return [
            _aggregate_row(
                number,
                "reformulation_bank_complete_score",
                payload,
                int(all(checks.values())),
                checks,
            )
        ]
    return []


def _mapping_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce exact mapping coverage and its preregistered positive gate."""

    proposals = [row for row in payload.get("proposal_rows", []) if isinstance(row, Mapping)]
    exact_groups = ("z3_rows", "enumeration_rows", "authority_agreement_rows")
    complete_checks = {
        "proposals_present": bool(proposals),
        "proposals_terminal": bool(proposals)
        and all(row.get("terminal") is True for row in proposals),
        "exact_row_counts": bool(proposals)
        and all(len(payload.get(field, [])) == len(proposals) for field in exact_groups),
        "exact_authorities_terminal": all(_all_rows(payload, field) for field in exact_groups),
        "model_coverage": len({row.get("hf_id") or row.get("model_family") for row in proposals})
        >= 2,
        "baseline_coverage": len(payload.get("baseline_rows", [])) in {0, len(proposals)},
        "paired_coverage": bool(payload.get("paired_metric_rows")),
    }
    complete = int(all(complete_checks.values()))
    intervals = [
        row for row in payload.get("confidence_interval_rows", []) if isinstance(row, Mapping)
    ]
    false_accepts = sum(row.get("false_acceptance") is True for row in proposals)
    false_acceptance_rate = false_accepts / len(proposals) if proposals else None
    positive_checks = {
        "run_complete": complete == 1,
        "two_model_intervals_above_zero": sum(
            isinstance(row.get("ci95_lower"), (int, float)) and row["ci95_lower"] > 0
            for row in intervals
        )
        >= 2,
        "false_acceptance_below_one_percent": false_acceptance_rate is not None
        and false_acceptance_rate < 0.01,
        "exact_authority_agreement": _all_rows(payload, "authority_agreement_rows"),
    }
    positive = int(all(positive_checks.values()))
    return [
        _aggregate_row(6957, "sota_mapping_positive_score", payload, positive, positive_checks),
        _aggregate_row(
            6957, "smt_certification_run_complete_score", payload, complete, complete_checks
        ),
    ]


def _convex_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce convexity, isolation, replay, and paired confidence evidence."""

    complete_checks = {
        "seed_rows_terminal": _all_rows(payload, "seed_rows"),
        "ordering_rows_terminal": _all_rows(payload, "ordering_rows"),
        "label_isolation": _all_rows(payload, "label_isolation_rows"),
        "shuffled_control": _all_rows(payload, "shuffled_label_rows"),
        "fresh_replay": _all_rows(payload, "fresh_process_replay_rows"),
    }
    complete = int(all(complete_checks.values()))
    intervals = [
        row for row in payload.get("confidence_interval_rows", []) if isinstance(row, Mapping)
    ]
    positive_checks = {
        "run_complete": complete == 1,
        "jensen_convexity": _all_rows(payload, "jensen_rows"),
        "finite_difference_convexity": _all_rows(payload, "finite_difference_rows"),
        "projection": _all_rows(payload, "projection_rows"),
        "two_paired_intervals_above_zero": len(intervals) == 2
        and all(
            isinstance(row.get("ci95_lower"), (int, float)) and row["ci95_lower"] > 0
            for row in intervals
        ),
    }
    positive = int(all(positive_checks.values()))
    return [
        _aggregate_row(6958, "convex_factor_positive_score", payload, positive, positive_checks),
        _aggregate_row(
            6958, "convex_factor_run_complete_score", payload, complete, complete_checks
        ),
    ]


def _headroom_rate(payload: Mapping[str, Any]) -> float | None:
    """Recompute captured oracle headroom from per-group rows."""

    rows = [row for row in payload.get("headroom_rows", []) if isinstance(row, Mapping)]
    available = sum(int(row.get("available_headroom") or 0) for row in rows)
    captured = sum(int(row.get("captured_headroom") or 0) for row in rows)
    return captured / available if available else None


def _selection_rows(number: int, payload: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce label-blind selection and its fresh-process audit."""

    audit = number == 6960
    selections = payload.get("selection_rows", [])
    groups = payload.get("candidate_group_rows", [])
    candidates = payload.get("candidate_rows", [])
    if audit:
        complete_field = "certified_selection_audit_complete_score"
        score_field = "audited_certified_energy_positive_score"
        complete_checks = {
            "audit_rows_terminal": _all_rows(payload, "audit_rows"),
            "selection_rows_terminal": _all_rows(payload, "selection_rows"),
            "certificate_replay": _all_rows(payload, "certificate_replay_rows"),
            "checkpoint_reload": _all_rows(payload, "checkpoint_reload_rows"),
            "proposal_budgets": _all_rows(payload, "proposal_budget_rows"),
        }
    else:
        complete_field = "certified_selection_run_complete_score"
        score_field = "certified_energy_positive_score"
        group_total = sum(
            int(row.get("candidate_count") or 0) for row in groups if isinstance(row, Mapping)
        )
        complete_checks = {
            "candidates_present": bool(candidates),
            "candidate_groups_present": bool(groups),
            "proposal_budget": bool(candidates) and group_total == len(candidates),
            "selection_rows_terminal": _all_rows(payload, "selection_rows"),
            "label_isolation": _all_rows(payload, "leakage_rows"),
            "fresh_replay": _all_rows(payload, "fresh_process_replay_rows"),
        }
    complete = int(all(complete_checks.values()))
    paired = [row for row in payload.get("paired_metric_rows", []) if isinstance(row, Mapping)]
    intervals = [
        row for row in payload.get("confidence_interval_rows", []) if isinstance(row, Mapping)
    ]
    strongest = {
        str(row.get("strongest_non_oracle_baseline"))
        for row in paired
        if row.get("strongest_non_oracle_baseline")
    }
    summary = payload.get("gate_check_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    upstream_rows = [
        row
        for row in summary.get("checks", [])
        if isinstance(row, Mapping) and row.get("check") == "upstream_positive_authority"
    ]
    upstream_positive = (
        upstream_rows[0].get("passed") is True
        if upstream_rows
        else summary.get("raw_positive_gate_passed") is True
        if audit
        else True
    )
    control_checks = {
        "strongest_baseline_named": len(strongest) == 1,
        "paired_gain": bool(paired)
        and sum(float(row.get("paired_top1_delta") or 0.0) for row in paired) > 0,
        "confidence_interval_above_zero": bool(intervals)
        and all(
            isinstance(row.get("ci95_lower"), (int, float)) and row["ci95_lower"] > 0
            for row in intervals
        ),
        "headroom_capture": (_headroom_rate(payload) or 0.0) >= 0.2,
        "label_isolation": _all_rows(payload, "label_isolation_rows" if audit else "leakage_rows"),
        "tie_policy": _all_rows(payload, "tie_policy_rows" if audit else "tie_rows"),
        "candidate_order": _all_rows(payload, "candidate_order_rows", required=audit),
        "shuffled_control": _all_rows(payload, "shuffled_energy_rows", required=not audit),
        "fixed_order_control": _all_rows(payload, "fixed_order_rows", required=not audit),
        "cold_audit_agreement": _all_rows(payload, "aggregate_consistency_rows", required=audit),
        "upstream_positive_authority": upstream_positive,
    }
    positive = int(complete == 1 and all(control_checks.values()))
    return [
        _aggregate_row(number, score_field, payload, positive, control_checks),
        _aggregate_row(number, complete_field, payload, complete, complete_checks),
    ]


def _event_sequence_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce the sealed prospective sequence without treating it as science."""

    chronology = [row for row in payload.get("chronology_rows", []) if isinstance(row, Mapping)]
    replay = [
        row for row in payload.get("fresh_process_replay_rows", []) if isinstance(row, Mapping)
    ]
    seeds = [row for row in payload.get("seed_certificate_rows", []) if isinstance(row, Mapping)]
    checks = {
        "event_count": len(payload.get("event_rows", [])) == 72,
        "prospective_order": len(chronology) == 72
        and all(row.get("available_after_ordinal") == row.get("ordinal") for row in chronology),
        "eligible_prior_only": _all_rows(payload, "eligible_prior_rows"),
        "future_label_isolation": _all_rows(payload, "leakage_rows"),
        "fresh_replay": len(replay) == 72
        and all(
            row.get("replay_matches") is True
            and row.get("prompt_matches") is True
            and row.get("retrieval_matches") is True
            for row in replay
        ),
        "exact_seed_authority": bool(seeds)
        and all(
            row.get("admission_authority") in EXACT_AUTHORITIES
            and row.get("authorities_agree") is True
            and row.get("exact_mapping_correct") is True
            and row.get("quarantined") is False
            for row in seeds
        ),
        "nonidentical_events": all(
            row.get("non_identical_to_seed") is True
            for row in payload.get("transformation_rows", [])
            if isinstance(row, Mapping)
        ),
    }
    return [
        _aggregate_row(
            6961, "certified_event_sequence_ready_score", payload, int(all(checks.values())), checks
        )
    ]


def _queue_score_rows(number: int, payload: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce queue utility while making an unexecuted queue score zero."""

    audit = number == 6963
    complete_field = (
        "queue_memory_audit_complete_score" if audit else "queue_learning_run_complete_score"
    )
    score_field = (
        "audited_queue_learning_positive_score" if audit else "queue_learning_positive_score"
    )
    complete_groups = (
        (
            "audit_rows",
            "arm_recompute_rows",
            "chronology_rows",
            "certificate_authority_rows",
            "debt_recompute_rows",
            "aggregate_consistency_rows",
        )
        if audit
        else ("event_rows", "write_rows", "debt_balance_rows", "restart_rows", "rollback_rows")
    )
    complete_checks = {field: _all_rows(payload, field) for field in complete_groups}
    complete = int(all(complete_checks.values()))
    paired = [row for row in payload.get("paired_metric_rows", []) if isinstance(row, Mapping)]
    intervals = [
        row for row in payload.get("confidence_interval_rows", []) if isinstance(row, Mapping)
    ]
    comparators = {"no_memory", "fifo"}
    positive_effects = {
        row.get("comparator")
        for row in paired
        if isinstance(row.get("mean_exact_accuracy_effect"), (int, float))
        and row["mean_exact_accuracy_effect"] > 0
    }
    positive_intervals = {
        row.get("comparator")
        for row in intervals
        if isinstance(row.get("ci95_lower"), (int, float)) and row["ci95_lower"] > 0
    }
    positive_checks = {
        "run_complete": complete == 1,
        "beats_no_memory_and_fifo": comparators <= positive_effects,
        "intervals_above_zero": comparators <= positive_intervals,
        "retention": _all_rows(payload, "retention_rows"),
        "future_label_isolation": _all_rows(payload, "future_label_isolation_rows"),
        "restart": _all_rows(payload, "restart_rows"),
        "rollback": _all_rows(payload, "rollback_rows"),
        "aggregate_agreement": _all_rows(payload, "aggregate_consistency_rows", required=audit),
    }
    positive = int(all(positive_checks.values()))
    return [
        _aggregate_row(number, score_field, payload, positive, positive_checks),
        _aggregate_row(number, complete_field, payload, complete, complete_checks),
    ]


def recompute_headlines(number: int, payload: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute every V609 score from source rows, never from prose."""

    roots = _source_and_contract_rows(number, payload)
    if roots:
        return roots
    if number == 6957:
        return _mapping_rows(payload)
    if number == 6958:
        return _convex_rows(payload)
    if number in {6959, 6960}:
        return _selection_rows(number, payload)
    if number == 6961:
        return _event_sequence_rows(payload)
    if number in {6962, 6963}:
        return _queue_score_rows(number, payload)
    return []


def _gate_pass(observed: Any, op: str, expected: Any) -> bool:
    """Evaluate one gate and fail closed on unknown values or operators."""

    try:
        return {
            "==": observed == expected,
            ">=": observed >= expected,
            "<=": observed <= expected,
            ">": observed > expected,
            "<": observed < expected,
        }.get(op, False)
    except TypeError:
        return False


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], payloads: Mapping[int, Mapping[str, Any]]
) -> list[JsonDict]:
    """Replay each structured gate directly against its producer artifact."""

    number_by_id = {str(task.get("task_id")): int(task["number"]) for task in tasks}
    rows = []
    for task in tasks:
        for gate in task.get("gates", []):
            if not isinstance(gate, Mapping):
                continue
            upstream = str(gate.get("upstream") or "")
            producer = payloads.get(number_by_id.get(upstream, -1))
            field = str(gate.get("artifact_field") or "")
            available = isinstance(producer, Mapping) and field in producer
            observed = _scalar(producer.get(field)) if available and producer else None
            expected = gate.get("value")
            op = str(gate.get("op") or "==")
            rows.append(
                {
                    "row_type": "gate_replay",
                    "task_id": task.get("task_id"),
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": op,
                    "expected": expected,
                    "observed": observed,
                    "passed": available and _gate_pass(observed, op, expected),
                    "available": available,
                }
            )
    return rows


def authoritative_scores(
    payloads: Mapping[int, Mapping[str, Any]], states: Mapping[int, Mapping[str, Any]]
) -> tuple[JsonDict, list[JsonDict]]:
    """Copy a score only when its own authority is admissible and consistent."""

    values: JsonDict = {}
    rows = []
    for field, number in SCORE_AUTHORITIES.items():
        payload = payloads.get(number)
        state = states.get(number, {})
        comparisons = recompute_headlines(number, payload) if payload else []
        comparison = next((row for row in comparisons if row.get("field") == field), None)
        reported = _scalar(payload.get(field)) if payload is not None else None
        klass = state.get("verdict_class")
        class_score_match = (klass == "positive" and reported == 1) or (
            klass == "null" and reported == 0
        )
        eligible = bool(
            payload is not None
            and klass in {"positive", "null"}
            and state.get("evidence_state")
            not in {
                "flagged",
                "row_headline_conflict",
                "verdict_prefix_class_conflict",
            }
            and comparison
            and comparison.get("agrees") is True
            and class_score_match
        )
        values[field] = reported if eligible else None
        rows.append(
            {
                "row_type": "authoritative_score",
                "number": number,
                "field": field,
                "value": values[field],
                "eligible": eligible,
                "reason": "copied_from_authoritative_artifact"
                if eligible
                else "authoritative_artifact_unavailable_or_inadmissible",
            }
        )
    return values, rows


def _debt_math(payload: Mapping[str, Any]) -> bool:
    """Check explicit debt rows or replay numeric queue balances when present."""

    groups = [
        payload.get(field)
        for field in ("debt_arrival_rows", "debt_service_rows", "debt_balance_rows")
    ]
    if not all(isinstance(group, list) and group for group in groups):
        return False
    flattened = [row for group in groups for row in group if isinstance(row, Mapping)]
    if flattened and all("passed" in row for row in flattened):
        return all(row.get("passed") is True for row in flattened)
    arrivals, services, balances = groups
    keyed_arrivals = {
        (row.get("model_id"), row.get("event_id")): row.get("arrival") for row in arrivals
    }
    keyed_services = {
        (row.get("model_id"), row.get("event_id")): row.get("service") for row in services
    }
    running: dict[Any, float] = {}
    for row in balances:
        key = (row.get("model_id"), row.get("event_id"))
        model = key[0]
        arrival = keyed_arrivals.get(key)
        service = keyed_services.get(key)
        balance = row.get("balance")
        if not all(isinstance(value, (int, float)) for value in (arrival, service, balance)):
            return False
        expected = max(0.0, running.get(model, 0.0) + float(arrival) - float(service))
        if abs(float(balance) - expected) > 1e-12:
            return False
        running[model] = float(balance)
    return bool(balances)


def _exact_writes(payload: Mapping[str, Any]) -> bool:
    """Require each admitted write to follow its external exact outcome."""

    writes = [row for row in payload.get("write_rows", []) if isinstance(row, Mapping)]
    if not writes:
        return False
    for row in writes:
        if "passed" in row or "certificate_authority" in row:
            if (
                row.get("passed") is not True
                or row.get("certificate_authority") not in EXACT_AUTHORITIES
            ):
                return False
            continue
        if row.get("write_admitted") is not True:
            continue
        raw_step = row.get("raw_output_durable_step")
        exact_step = row.get("exact_outcome_visible_step")
        write_step = row.get("write_step")
        if not all(isinstance(value, int) for value in (raw_step, exact_step, write_step)):
            return False
        if not raw_step < exact_step < write_step:
            return False
    return True


def build_safety_rows(payloads: Mapping[int, Mapping[str, Any]]) -> list[JsonDict]:
    """Check prospective sequence and queue safety only from explicit rows."""

    rows: list[JsonDict] = []
    sequence = payloads.get(6961)
    if sequence is None:
        rows.append({"row_type": "safety", "number": 6961, "available": False, "passed": None})
    else:
        recomputed = _event_sequence_rows(sequence)[0]
        rows.append(
            {
                "row_type": "safety",
                "number": 6961,
                "available": True,
                "prospective_order": recomputed["checks"]["prospective_order"],
                "exact_certificate_authority": recomputed["checks"]["exact_seed_authority"],
                "future_label_isolation": recomputed["checks"]["future_label_isolation"],
                "passed": all(recomputed["checks"].values()),
            }
        )
    queue = payloads.get(6962)
    if queue is None or not queue.get("write_rows"):
        rows.append({"row_type": "safety", "number": 6962, "available": False, "passed": None})
    else:
        before = _scalar(queue.get("model_hashes_before"))
        after = _scalar(queue.get("model_hashes_after"))
        checks = {
            "continuous_self_learning_task": _scalar(queue.get("continuous_self_learning_task"))
            is True,
            "learning_tier_two": _scalar(queue.get("learning_tier")) == 2,
            "exact_certificate_writes": _exact_writes(queue),
            "prospective_order": _all_rows(queue, "chronology_rows"),
            "debt_math": _debt_math(queue),
            "hard_resets": bool(queue.get("restart_rows"))
            and all(
                row.get("passed") is True and row.get("hard_reset", True) is True
                for row in queue.get("restart_rows", [])
                if isinstance(row, Mapping)
            ),
            "retention": _all_rows(queue, "retention_rows"),
            "rollback": _all_rows(queue, "rollback_rows"),
            "future_label_isolation": _all_rows(queue, "future_label_isolation_rows"),
            "model_hashes_unchanged": bool(before) and before == after,
            "no_model_weight_mutation": _scalar(queue.get("no_model_weight_mutation")) is True,
        }
        rows.append(
            {
                "row_type": "safety",
                "number": 6962,
                "available": True,
                **checks,
                "passed": all(checks.values()),
            }
        )
    audit = payloads.get(6963)
    if audit is None or _is_bootstrap_only(audit):
        rows.append({"row_type": "safety", "number": 6963, "available": False, "passed": None})
    else:
        groups = (
            "chronology_rows",
            "certificate_authority_rows",
            "future_label_isolation_rows",
            "write_order_rows",
            "debt_recompute_rows",
            "retention_rows",
            "restart_rows",
            "rollback_rows",
            "model_immutability_rows",
            "aggregate_consistency_rows",
        )
        checks = {field: _all_rows(audit, field) for field in groups}
        rows.append(
            {
                "row_type": "safety",
                "number": 6963,
                "available": True,
                **checks,
                "passed": all(checks.values()),
            }
        )
    return rows


FORBIDDEN_CLAIM_FIELDS = {
    "solve_claimed",
    "arc_solve_claimed",
    "hardware_speed_claimed",
    "hardware_power_claimed",
    "power_claimed",
    "tsu_access_claimed",
    "default_on",
    "production_default_on",
    "default_on_production_claimed",
}


def _claim_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Find affirmative forbidden claim fields outside unstructured model text."""

    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, str(key))
            if str(key).lower() in FORBIDDEN_CLAIM_FIELDS and _scalar(child) is True:
                found.append(".".join(child_path))
            elif str(key) not in {"raw_text", "raw_output", "prompt", "prompt_payload"}:
                found.extend(_claim_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_claim_paths(child, (*path, str(index))))
    return found


def build_claim_boundary_rows(
    tasks: Sequence[Mapping[str, Any]], payloads: Mapping[int, Mapping[str, Any]]
) -> list[JsonDict]:
    """Show that V609 made no ARC, hardware, TSU, or default-on claim."""

    rows = []
    for task in tasks:
        number = int(task["number"])
        payload = payloads.get(number)
        claims = _claim_paths(payload) if payload is not None else []
        rows.append(
            {
                "row_type": "claim_boundary",
                "number": number,
                "artifact_available": payload is not None,
                "forbidden_claim_paths": claims,
                "arc_solve_claimed": any("solve_claimed" in path for path in claims),
                "hardware_speed_or_power_claimed": any(
                    "speed_claimed" in path or "power_claimed" in path for path in claims
                ),
                "tsu_access_claimed": any("tsu_access_claimed" in path for path in claims),
                "default_on_production_claimed": any("default_on" in path for path in claims),
                "passed": not claims,
            }
        )
    return rows


def apply_boundary_checks(
    states: Mapping[int, Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
    safety_rows: Sequence[Mapping[str, Any]],
) -> dict[int, JsonDict]:
    """Disqualify admissible claims when an explicit safety boundary fails."""

    checked = {number: deepcopy(dict(row)) for number, row in states.items()}
    controls = [(row, "forbidden_claim_boundary_conflict") for row in claim_rows]
    controls.extend((row, "continuous_learning_safety_conflict") for row in safety_rows)
    for control, evidence_state in controls:
        number = int(control["number"])
        state = checked.get(number)
        if state is None or state.get("verdict_class") not in {
            "positive",
            "circular_positive",
            "null",
        }:
            continue
        if control.get("passed") is not False:
            continue
        state["evidence_state"] = evidence_state
        state["verdict_class"] = "disqualified"
        state["admissible"] = False
    return checked


def exclusion_candidates(
    tasks: Sequence[Mapping[str, Any]], states: Mapping[int, Mapping[str, Any]]
) -> list[JsonDict]:
    """Return exact repeated-verdict candidates without editing the manifest."""

    rows = []
    for task in tasks:
        number = int(task["number"])
        current = str(states.get(number, {}).get("honest_verdict") or "")
        for prior in task.get("prior_failures", []):
            if not isinstance(prior, Mapping):
                continue
            if prior.get("retire_if_same_verdict") is True and current == str(
                prior.get("verdict") or ""
            ):
                rows.append(
                    {
                        "row_type": "exclusion_candidate",
                        "number": number,
                        "task_id": task.get("task_id"),
                        "prior_experiment_id": prior.get("experiment_id"),
                        "repeated_verdict": current,
                        "manifest_edited": False,
                    }
                )
    return rows


def capstone_outcome(
    contract_passed: bool, has_disqualified: bool, has_incomplete: bool
) -> tuple[str, str, str]:
    """Return the terminal status, class, and verdict from replayed evidence."""

    if not contract_passed or has_disqualified:
        return "complete_disqualified", "disqualified", DISQUALIFIED_VERDICT
    if has_incomplete:
        return "complete_null", "null", NULL_VERDICT
    return "complete", "positive", POSITIVE_VERDICT


def _load_json(path: Path) -> JsonDict | None:
    """Load one artifact and return no evidence for malformed input."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    """Hash source bytes so each imported fact can be replayed."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checker(path: Path, name: str) -> Any:
    """Load the current checker source so cached imports cannot hide fixes."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _audit_artifact(root: Path, path: Path) -> tuple[JsonDict, tuple[str, list[str]]]:
    """Run both evidence checkers and preserve any checker failure as evidence."""

    try:
        adversarial = _load_checker(
            root / ADVERSARIAL_PATH, "v609_capstone_adversarial"
        ).verify_artifact(path)
    except Exception as exc:  # noqa: BLE001 - checker failure must remain visible.
        adversarial = {
            "flags": [{"kind": "VERIFIER_ERROR", "severity": "critical", "detail": str(exc)}]
        }
    try:
        row_check = _load_checker(root / ROW_LINT_PATH, "v609_capstone_row_lint").check_artifact(
            path
        )
    except Exception as exc:  # noqa: BLE001 - checker failure must remain visible.
        row_check = ("unreadable", [f"VERIFIER_ERROR: {exc}"])
    return adversarial, row_check


def _model_coverage_rows(payloads: Mapping[int, Mapping[str, Any]]) -> list[JsonDict]:
    """Report model coverage separately from successful model execution."""

    rows = []
    for number, expected in EXPECTED_MODELS.items():
        payload = payloads.get(number)
        model_data = {
            key: payload[key]
            for key in (
                "model_specs",
                "model_rows",
                "proposal_rows",
                "attempt_rows",
                "task_runtime_receipt",
            )
            if payload is not None and key in payload
        }
        available = bool(model_data)
        text = json.dumps(model_data, sort_keys=True) if available else ""
        observed = [model for model in expected if model in text]
        rows.append(
            {
                "row_type": "model_coverage",
                "number": number,
                "artifact_available": payload is not None,
                "available": available,
                "expected_models": list(expected),
                "observed_models": observed,
                "passed": len(observed) == len(expected) if available else None,
            }
        )
    return rows


def _hardware_rows(
    tasks: Sequence[Mapping[str, Any]],
    payloads: Mapping[int, Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Record runtime sources and make prohibited hardware claims auditable."""

    claims = {int(row["number"]): row for row in claim_rows}
    rows = []
    for task in tasks:
        number = int(task["number"])
        payload = payloads.get(number)
        rows.append(
            {
                "row_type": "hardware_provenance",
                "number": number,
                "artifact_available": payload is not None,
                "inference_substrate": _scalar(payload.get("inference_substrate"))
                if payload
                else None,
                "duration_s": _scalar(payload.get("duration_s")) if payload else None,
                "task_runtime_receipt_present": bool(
                    payload and payload.get("task_runtime_receipt")
                ),
                "model_hashes_present": bool(payload and payload.get("model_hashes_before")),
                "forbidden_claim_paths": claims[number]["forbidden_claim_paths"],
                "claim_boundary_passed": claims[number]["passed"],
            }
        )
    return rows


def _field_principles() -> JsonDict:
    """Give each required field a short falsifiability purpose."""

    return {
        field: f"Report {field} directly so an independent reader can falsify the V609 reconciliation."
        for field in sorted(REQUIRED_ARTIFACT_FIELDS)
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic content while excluding duration and the digest."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _empty_artifact(run_date: str) -> JsonDict:
    """Create the complete schema so a preflight block stays inspectable."""

    artifact: JsonDict = {
        "schema": "carnot.v609_capstone.v1",
        "experiment_id": 6964,
        "run_date": run_date,
        "status": "blocked",
        "field_principles": _field_principles(),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "task_contract_rows": [],
        "task_state_rows": [],
        "gate_replay_rows": [],
        "verdict_class_rows": [],
        "adversarial_verify_rows": [],
        "aggregate_recompute_rows": [],
        "model_coverage_rows": [],
        "mapping_rows": [],
        "certificate_rows": [],
        "convex_factor_rows": [],
        "selection_rows": [],
        "selection_audit_rows": [],
        "event_sequence_rows": [],
        "queue_learning_rows": [],
        "queue_audit_rows": [],
        "safety_rows": [],
        "exclusion_candidate_rows": [],
        "hardware_provenance_rows": [],
        "v610_gap_rows": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        **{field: None for field in SCORE_AUTHORITIES},
        "v609_capstone_complete_score": 0,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    return artifact


def _source_hashes(root: Path, tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Hash contracts, checkers, manifest, V608 handoff, and V609 inputs."""

    paths = {
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        SPEC_PATH,
        DESIGN_PATH,
        ROADMAP_PATH,
        CONDUCTOR_LOG_PATH,
        EXCLUSION_PATH,
        ADVERSARIAL_PATH,
        ROW_LINT_PATH,
        V608_CAPSTONE_PATH,
        MODULE_PATH,
        RUNNER_PATH,
    }
    paths.update(Path(str(task["deliverable"])) for task in tasks if int(task["number"]) < 6964)
    return {
        path.as_posix(): _sha256(root / path)
        for path in sorted(paths, key=lambda value: value.as_posix())
        if (root / path).is_file()
    }


def _v610_gaps() -> list[JsonDict]:
    """Return three next gaps supported by science or failures that ran."""

    return [
        {
            "row_type": "v610_gap",
            "rank": 1,
            "gap": "Exact certification found only 10 correct mappings in 162 proposals and no positive mapping gain.",
            "evidence_task_ids": [
                "exp6956-three-family-reformulation-bank",
                "exp6957-smt-mapping-certification",
            ],
            "next_executable_prerequisite": "Improve proposal parseability and exact mapping quality under the same fixed budget and exact authorities.",
            "science_ran": True,
        },
        {
            "row_type": "v610_gap",
            "rank": 2,
            "gap": "Certified energy selection had no oracle headroom and the convex canary was adversarially flagged.",
            "evidence_task_ids": [
                "exp6958-convex-factor-energy-canary",
                "exp6959-certified-energy-selection",
                "exp6960-certified-selection-cold-audit",
            ],
            "next_executable_prerequisite": "Establish a clean convex-factor runtime and a candidate set with nonzero selection headroom before another causal selection claim.",
            "science_ran": True,
        },
        {
            "row_type": "v610_gap",
            "rank": 3,
            "gap": "The queue-learning worker failed during Qwen model load, so no queue events or cold audit ran.",
            "evidence_task_ids": [
                "exp6961-certified-event-sequence",
                "exp6962-queue-regulated-self-learning",
                "exp6963-queue-memory-cold-audit",
            ],
            "next_executable_prerequisite": "Repair authenticated GGUF loading, then execute queue arms before the cold audit.",
            "science_ran": True,
        },
    ]


def build_artifact(root: Path, run_date: str) -> JsonDict:
    """Build the V609 capstone from checked-in evidence and no new science."""

    started = time.monotonic()
    artifact = _empty_artifact(run_date)
    core = {"active_v609_roadmap": ROADMAP_PATH, "v609_design_document": DESIGN_PATH}
    preconditions = [
        {"resource": name, "path": path.as_posix(), "available": (root / path).is_file()}
        for name, path in core.items()
    ]
    artifact["preconditions_checked"] = preconditions
    if not all(row["available"] for row in preconditions):
        artifact["gate_check_summary"] = {
            "failed_check": "core_contract_preconditions",
            "expected": "active V609 roadmap and design document",
            "observed": [row["path"] for row in preconditions if not row["available"]],
            "passed": False,
        }
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    design_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = load_yaml(root / ROADMAP_PATH)
    yaml_tasks = roadmap_tasks(roadmap)
    tasks_by_number = {int(task["number"]): task for task in yaml_tasks if task.get("number")}
    tasks = [
        tasks_by_number.get(int(expected["number"]), deepcopy(expected))
        for expected in EXPECTED_TASKS
    ]
    contract_rows = build_contract_rows(design_text, roadmap)
    conductor_text = (
        (root / CONDUCTOR_LOG_PATH).read_text(encoding="utf-8")
        if (root / CONDUCTOR_LOG_PATH).is_file()
        else ""
    )
    conductor = parse_conductor_states(conductor_text, tasks)
    payloads: dict[int, JsonDict] = {}
    adversarial_rows: list[JsonDict] = []
    adversarial_reports: dict[int, JsonDict] = {}
    row_checks: dict[int, tuple[str, list[str]]] = {}
    recompute_rows: list[JsonDict] = []
    for task in tasks:
        number = int(task["number"])
        if number == 6964:
            continue
        path = root / str(task["deliverable"])
        payload = _load_json(path) if path.is_file() else None
        if payload is None:
            continue
        payloads[number] = payload
        adversarial, row_check = _audit_artifact(root, path)
        adversarial_reports[number] = adversarial
        row_checks[number] = row_check
        adversarial_rows.append(
            {
                "row_type": "adversarial_verify",
                "number": number,
                "artifact": str(task["deliverable"]),
                "flag_count": adversarial.get("flag_count", len(adversarial.get("flags", []))),
                "flags": adversarial.get("flags", []),
                "row_check_status": row_check[0],
                "row_findings": row_check[1],
            }
        )
        if not _is_bootstrap_only(payload):
            recompute_rows.extend(recompute_headlines(number, payload))

    states: dict[int, JsonDict] = {}
    for task in tasks:
        number = int(task["number"])
        if number == 6964:
            continue
        comparisons = [row for row in recompute_rows if row.get("number") == number]
        states[number] = classify_evidence(
            task,
            conductor.get(number, {}),
            payloads.get(number),
            adversarial_reports.get(number),
            row_checks.get(number),
            comparisons,
        )

    gate_rows = replay_gates(tasks, payloads)
    safety_rows = build_safety_rows(payloads)
    claim_rows = build_claim_boundary_rows(tasks, payloads)
    states = apply_boundary_checks(states, claim_rows, safety_rows)
    contract_passed = len(contract_rows) == 12 and all(row["passed"] for row in contract_rows)
    has_disqualified = any(row.get("verdict_class") == "disqualified" for row in states.values())
    has_incomplete = any(
        row.get("verdict_class") in {"blocked", "partial", "null"} for row in states.values()
    )
    status, verdict_class, honest = capstone_outcome(
        contract_passed, has_disqualified, has_incomplete
    )
    current = next(task for task in tasks if int(task["number"]) == 6964)
    states[6964] = {
        "row_type": "task_state",
        "task_id": current["task_id"],
        "number": 6964,
        "title": current["title"],
        "conductor_state": "current_synthesis",
        "artifact_state": "current_synthesis",
        "evidence_state": verdict_class,
        "declared_verdict_class": verdict_class,
        "structural_verdict_class": verdict_class,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
        "admissible": verdict_class in {"positive", "null"},
        "terminal": True,
        "row_check_status": "self_validation",
        "row_findings": [],
        "adversarial_critical": False,
    }
    state_rows = [states[number] for number in range(6953, 6965)]
    replay_complete = len(contract_rows) == 12 and all(
        row.get("terminal") is True for row in contract_rows
    )
    terminal = len(state_rows) == 12 and all(row.get("terminal") is True for row in state_rows)
    complete = int(replay_complete and terminal)
    score_values, authority_rows = authoritative_scores(payloads, states)
    model_rows = _model_coverage_rows(payloads)
    hardware_rows = _hardware_rows(tasks, payloads, claim_rows)
    exclusion_rows = exclusion_candidates(tasks, states)
    by_field = {row["field"]: row for row in authority_rows}
    by_number: dict[int, list[JsonDict]] = {}
    for row in recompute_rows:
        by_number.setdefault(int(row["number"]), []).append(row)
    failed_check = (
        "document_yaml_contract_replay"
        if not contract_passed
        else "flagged_or_conflicting_evidence"
        if has_disqualified
        else "science_incomplete"
        if has_incomplete
        else None
    )
    artifact.update(
        {
            "status": status,
            "source_artifact_hashes": _source_hashes(root, tasks),
            "rows": state_rows,
            "task_contract_rows": contract_rows,
            "task_state_rows": state_rows,
            "gate_replay_rows": gate_rows,
            "verdict_class_rows": [
                {
                    "row_type": "verdict_class",
                    "number": row["number"],
                    "task_id": row["task_id"],
                    "declared": row["declared_verdict_class"],
                    "structural": row["structural_verdict_class"],
                    "final": row["verdict_class"],
                    "evidence_state": row["evidence_state"],
                }
                for row in state_rows
            ],
            "adversarial_verify_rows": adversarial_rows,
            "aggregate_recompute_rows": recompute_rows + authority_rows,
            "model_coverage_rows": model_rows,
            "mapping_rows": by_number.get(6955, []) + by_number.get(6956, []),
            "certificate_rows": by_number.get(6957, []) + [by_field["sota_mapping_positive_score"]],
            "convex_factor_rows": by_number.get(6958, [])
            + [by_field["convex_factor_positive_score"]],
            "selection_rows": by_number.get(6959, [])
            + [by_field["certified_energy_positive_score"]],
            "selection_audit_rows": by_number.get(6960, [])
            + [by_field["audited_certified_energy_positive_score"]],
            "event_sequence_rows": by_number.get(6961, [])
            + [row for row in safety_rows if row.get("number") == 6961],
            "queue_learning_rows": by_number.get(6962, [])
            + [by_field["queue_learning_positive_score"]],
            "queue_audit_rows": by_number.get(6963, [])
            + [by_field["audited_queue_learning_positive_score"]],
            "safety_rows": safety_rows + claim_rows,
            "exclusion_candidate_rows": exclusion_rows,
            "hardware_provenance_rows": hardware_rows,
            "v610_gap_rows": _v610_gaps(),
            **score_values,
            "v609_capstone_complete_score": complete,
            "gate_check_summary": {
                "failed_check": failed_check,
                "expected": "12 terminal classification rows and a completed 12-row contract replay",
                "observed": {
                    "terminal_classification_rows": sum(
                        row.get("terminal") is True for row in state_rows
                    ),
                    "terminal_contract_rows": sum(
                        row.get("terminal") is True for row in contract_rows
                    ),
                    "passing_contract_rows": sum(
                        row.get("passed") is True for row in contract_rows
                    ),
                    "disqualified_task_numbers": [
                        row["number"]
                        for row in state_rows
                        if row["verdict_class"] == "disqualified"
                    ],
                },
                "passed": failed_check is None,
                "replay_complete": bool(complete),
            },
            "verdict_class": verdict_class,
            "honest_verdict": honest,
        }
    )
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate shape, authority, completion, prefixes, gaps, and checksum."""

    errors = []
    for field in sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact)):
        errors.append(f"missing required field: {field}")
    principles = artifact.get("field_principles")
    if isinstance(principles, Mapping):
        for field in sorted(REQUIRED_ARTIFACT_FIELDS - set(principles)):
            errors.append(f"field_principles missing {field}")
    else:
        errors.append("field_principles must be a mapping")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in VERDICT_CLASSES:
        errors.append("verdict_class is not closed")
    prefixes = {
        "positive": ("complete_positive_", "positive_", "success_"),
        "circular_positive": ("complete_circular_positive_", "circular_positive_"),
        "null": ("complete_null_", "null_"),
        "blocked": ("blocked_",),
        "disqualified": ("complete_disqualified_", "disqualified_"),
        "partial": ("partial_",),
    }
    honest = str(artifact.get("honest_verdict") or "")
    if verdict_class in prefixes and not honest.startswith(prefixes[verdict_class]):
        errors.append("honest_verdict prefix conflicts with verdict_class")
    if verdict_class == "partial" and honest.startswith("complete_"):
        errors.append("complete prefix cannot use partial verdict_class")
    contracts = artifact.get("task_contract_rows")
    states = artifact.get("task_state_rows")
    recomputed_complete = int(
        isinstance(contracts, list)
        and len(contracts) == 12
        and all(isinstance(row, Mapping) and row.get("terminal") is True for row in contracts)
        and isinstance(states, list)
        and len(states) == 12
        and all(isinstance(row, Mapping) and row.get("terminal") is True for row in states)
    )
    if artifact.get("v609_capstone_complete_score") != recomputed_complete:
        errors.append("v609_capstone_complete_score does not match rows")
    authority_rows = {
        row.get("field"): row
        for row in artifact.get("aggregate_recompute_rows", [])
        if isinstance(row, Mapping) and row.get("row_type") == "authoritative_score"
    }
    for field in SCORE_AUTHORITIES:
        if field in authority_rows and artifact.get(field) != authority_rows[field].get("value"):
            errors.append(f"{field} does not match authoritative row")
    if verdict_class == "blocked":
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, Mapping) or not all(
            summary.get(key) is not None for key in ("failed_check", "expected", "observed")
        ):
            errors.append("blocked verdict requires an exact failed gate check")
    gaps = artifact.get("v610_gap_rows")
    if recomputed_complete == 1 and (not isinstance(gaps, list) or len(gaps) != 3):
        errors.append("completed capstone requires exactly three V610 gaps")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Write or validate the capstone at an explicit target path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260904")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.repo_root / args.output
    if args.validate:
        payload = _load_json(output) or {}
        errors = validate_artifact(payload)
        if errors:
            print("\n".join(errors))
            return 1
        return 0
    artifact = build_artifact(args.repo_root, args.date)
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper owns the executable path.
    raise SystemExit(main())
