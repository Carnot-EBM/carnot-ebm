"""Reconcile V611 contracts and primary evidence without rerunning science.

The reducer reads each task artifact separately. It derives headline values
from unit rows because a copied aggregate can preserve a producer error. See
REQ-REPORT-6983.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6983_v611_capstone.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")
MILESTONE = "2026.09.611"
INFERENCE_SUBSTRATE = "deterministic_independent_artifact_reconciliation"
RANDOM_SEED = 6983
BLOCKED_VERDICT = "blocked_v611_capstone"
NULL_VERDICT = "complete_null_v611_capstone_no_non_circular_science_positive"
POSITIVE_VERDICT = "complete_positive_v611_capstone_row_backed_science_positive"
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
BRANCH_STATES = {
    "absent",
    "pre_gate_blocked",
    "artifact_blocked",
    "null",
    "positive",
    "circular_positive",
    "disqualified",
    "partial",
}
REQUIRED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SCIENCE_TASKS = {6976, 6977, 6978, 6980, 6981, 6982}


def _gate(upstream: str, artifact_field: str, op: str = "==", value: Any = 1) -> JsonDict:
    """Return one normalized dependency gate."""

    return {
        "upstream": upstream,
        "artifact_field": artifact_field,
        "op": op,
        "value": value,
    }


EXPECTED_TASKS: tuple[JsonDict, ...] = (
    {
        "number": 6972,
        "task_id": "exp6972-v611-contract-advisory",
        "title": "V611 advisory task-contract and gate audit",
        "deliverable": "results/experiment_6972_v611_contract_advisory.json",
        "gates": [],
    },
    {
        "number": 6973,
        "task_id": "exp6973-lease-aware-gguf-runtime",
        "title": "Lease-aware three-family GGUF runtime handoff",
        "deliverable": "results/experiment_6973_lease_aware_gguf_runtime.json",
        "gates": [],
    },
    {
        "number": 6974,
        "task_id": "exp6974-claim-provenance-duration-lint",
        "title": "Claim-provenance-aware duration verification",
        "deliverable": "results/experiment_6974_claim_provenance_duration_lint.json",
        "gates": [],
    },
    {
        "number": 6975,
        "task_id": "exp6975-delayed-constraint-candidate-bank",
        "title": "Three-family delayed-constraint candidate bank",
        "deliverable": "results/experiment_6975_delayed_constraint_candidate_bank.json",
        "gates": [
            _gate("exp6973-lease-aware-gguf-runtime", "lease_aware_runtime_ready_score"),
            _gate(
                "exp6974-claim-provenance-duration-lint",
                "fixture_admissibility_ready_score",
            ),
        ],
    },
    {
        "number": 6976,
        "task_id": "exp6976-exact-candidate-certification",
        "title": "Exact candidate certification and policy selection",
        "deliverable": "results/experiment_6976_exact_candidate_certification.json",
        "gates": [
            _gate(
                "exp6975-delayed-constraint-candidate-bank",
                "candidate_bank_complete_score",
            )
        ],
    },
    {
        "number": 6977,
        "task_id": "exp6977-certified-pwa-kan-energy",
        "title": "Certified PWA-KAN residual energy",
        "deliverable": "results/experiment_6977_certified_pwa_kan_energy.json",
        "gates": [
            _gate(
                "exp6976-exact-candidate-certification",
                "candidate_certification_complete_score",
            )
        ],
    },
    {
        "number": 6978,
        "task_id": "exp6978-transactional-constraint-self-learning",
        "title": "Transactional verifier-grounded continuous self-learning",
        "deliverable": "results/experiment_6978_transactional_constraint_self_learning.json",
        "gates": [
            _gate("exp6973-lease-aware-gguf-runtime", "lease_aware_runtime_ready_score"),
            _gate(
                "exp6974-claim-provenance-duration-lint",
                "fixture_admissibility_ready_score",
            ),
            _gate(
                "exp6976-exact-candidate-certification",
                "selected_policy_ready_score",
            ),
        ],
    },
    {
        "number": 6979,
        "task_id": "exp6979-self-learning-cold-audit",
        "title": "Fresh-process self-learning safety audit",
        "deliverable": "results/experiment_6979_self_learning_cold_audit.json",
        "gates": [
            _gate(
                "exp6978-transactional-constraint-self-learning",
                "self_learning_run_complete_score",
            )
        ],
    },
    {
        "number": 6980,
        "task_id": "exp6980-spilled-energy-requalification",
        "title": "Span-localized spilled-energy requalification",
        "deliverable": "results/experiment_6980_spilled_energy_requalification.json",
        "gates": [
            _gate(
                "exp6975-delayed-constraint-candidate-bank",
                "candidate_bank_complete_score",
            ),
            _gate(
                "exp6976-exact-candidate-certification",
                "candidate_certification_complete_score",
            ),
        ],
    },
    {
        "number": 6981,
        "task_id": "exp6981-arc-live-engine-generalization-audit",
        "title": "ARC live-engine reachability and generalization audit",
        "deliverable": "results/experiment_6981_arc_live_engine_generalization_audit.json",
        "gates": [],
    },
    {
        "number": 6982,
        "task_id": "exp6982-hard-feasible-hybrid-selection",
        "title": "Hard-feasible hybrid energy selection",
        "deliverable": "results/experiment_6982_hard_feasible_hybrid_selection.json",
        "gates": [
            _gate(
                "exp6976-exact-candidate-certification",
                "candidate_certification_complete_score",
            ),
            _gate(
                "exp6976-exact-candidate-certification",
                "heldout_headroom_group_count",
                ">=",
                2,
            ),
            _gate(
                "exp6977-certified-pwa-kan-energy",
                "certified_pwa_energy_ready_score",
            ),
        ],
    },
    {
        "number": 6983,
        "task_id": "exp6983-v611-capstone",
        "title": "V611 independent capstone and V612 handoff",
        "deliverable": OUTPUT_PATH.as_posix(),
        "gates": [],
    },
)
EXPECTED_TASK_IDS = [str(task["task_id"]) for task in EXPECTED_TASKS]

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
    "expected_task_ids",
    "observed_task_ids",
    "document_contract_rows",
    "yaml_contract_rows",
    "gate_contract_rows",
    "rows",
    "per_task_results",
    "artifact_presence_rows",
    "flagged_exclusion_rows",
    "pre_gate_rows",
    "verdict_class_rows",
    "branch_status_rows",
    "runtime_recomputation",
    "constraint_recomputation",
    "pwa_certificate_recomputation",
    "self_learning_recomputation",
    "spilled_energy_recomputation",
    "arc_recomputation",
    "hybrid_selection_recomputation",
    "prior_failure_disposition_rows",
    "blocked_cause_rows",
    "command_receipt_rows",
    "v612_handoff_rows",
    "v611_capstone_complete_score",
    "v611_task_contract_conforms_score",
    "v611_science_positive_score",
    "paper_ready",
    "paper_ready_source",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}


def spec_anchors(text: str) -> list[str]:
    """Return requirement anchors used by focused traceability tests."""

    return re.findall(r"(?:REQ|SCENARIO)-[A-Z0-9-]+", text)


def load_yaml(path: Path) -> JsonDict:
    """Load one YAML mapping and reject a non-mapping root."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return value


def _number(value: Any) -> int | None:
    """Read the numeric prefix of a full experiment task ID."""

    match = re.match(r"exp(\d+)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _design_milestone(text: str) -> str:
    """Read the milestone from the document without consulting YAML."""

    match = re.search(r"\*\*Milestone:\*\*\s*([^\s]+)", text)
    return match.group(1) if match else ""


def _parse_design_gates(text: str) -> list[JsonDict]:
    """Parse a design-table gate cell without using roadmap values."""

    if text.strip().lower() == "none":
        return []
    task_by_number = {int(task["number"]): str(task["task_id"]) for task in EXPECTED_TASKS}
    gates = []
    for item in text.split(";"):
        clean = item.strip().replace("`", "")
        match = re.fullmatch(
            r"(exp\d+[a-z0-9-]*)\s+([A-Za-z0-9_]+)\s*(==|>=|<=|>|<)\s*(.+)",
            clean,
        )
        if not match:
            gates.append({"unparsed": item.strip()})
            continue
        number = _number(match.group(1))
        upstream = task_by_number.get(number or -1, match.group(1))
        gates.append(_gate(upstream, match.group(2), match.group(3), yaml.safe_load(match.group(4))))
    return gates


def parse_document_contract(text: str) -> list[JsonDict]:
    """Parse the Markdown task table as an independent contract."""

    marker = re.search(r"^## Exact task contract\s*$", text, re.MULTILINE | re.IGNORECASE)
    if marker is None:
        return []
    section = text[marker.end() :]
    next_heading = re.search(r"^##\s+", section, re.MULTILINE)
    if next_heading is not None:
        section = section[: next_heading.start()]
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*(exp\d+[a-z0-9-]*)\s*\|\s*([^|]+?)\s*"
        r"\|\s*`?([^|`]+?)`?\s*\|\s*([^|]+?)\s*\|$",
        re.MULTILINE,
    )
    milestone = _design_milestone(text)
    return [
        {
            "source": "document",
            "order": int(order),
            "number": _number(task_id),
            "task_id": task_id.strip(),
            "title": title.strip(),
            "deliverable": deliverable.strip(),
            "milestone": milestone,
            "gates": _parse_design_gates(gates),
            "passed": True,
        }
        for order, task_id, title, deliverable, gates in pattern.findall(section)
    ]


def parse_yaml_contract(roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Normalize YAML tasks without borrowing document values."""

    raw_tasks = roadmap.get("tasks")
    if not isinstance(raw_tasks, list):
        return []
    milestone = str(roadmap.get("milestone") or "")
    rows = []
    for order, raw in enumerate(raw_tasks, 1):
        if not isinstance(raw, Mapping):
            rows.append(
                {
                    "source": "yaml",
                    "order": order,
                    "number": None,
                    "task_id": "",
                    "title": "",
                    "deliverable": "",
                    "milestone": milestone,
                    "gates": [],
                    "prompt": "",
                    "prior_failures": [],
                    "passed": False,
                }
            )
            continue
        rows.append(
            {
                "source": "yaml",
                "order": order,
                "number": _number(raw.get("id")),
                "task_id": str(raw.get("id") or ""),
                "title": str(raw.get("title") or ""),
                "deliverable": str(raw.get("deliverable") or ""),
                "milestone": str(raw.get("milestone") or milestone),
                "gates": deepcopy(raw.get("gated_on") or []),
                "prompt": str(raw.get("prompt") or ""),
                "prior_failures": deepcopy(raw.get("prior_failures") or []),
                "passed": True,
            }
        )
    return rows


def _expected_row_matches(row: Mapping[str, Any], expected: Mapping[str, Any], order: int) -> bool:
    """Check one source row against the fixed 12-task contract."""

    return all(
        (
            row.get("order") == order,
            row.get("number") == expected["number"],
            row.get("task_id") == expected["task_id"],
            row.get("title") == expected["title"],
            row.get("deliverable") == expected["deliverable"],
            row.get("milestone") == MILESTONE,
            row.get("gates") == expected["gates"],
        )
    )


def build_gate_contract_rows(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Compare each dependency and confirm its producer field exists."""

    document_by_id = {str(row.get("task_id")): row for row in document_rows}
    yaml_by_id = {str(row.get("task_id")): row for row in yaml_rows}
    order_by_id = {str(row.get("task_id")): int(row.get("order") or 0) for row in yaml_rows}
    rows = []
    for expected in EXPECTED_TASKS:
        task_id = str(expected["task_id"])
        doc_gates = document_by_id.get(task_id, {}).get("gates", [])
        yaml_gates = yaml_by_id.get(task_id, {}).get("gates", [])
        for index, expected_gate in enumerate(expected["gates"]):
            document_gate = doc_gates[index] if index < len(doc_gates) else None
            yaml_gate = yaml_gates[index] if index < len(yaml_gates) else None
            upstream = str(expected_gate["upstream"])
            producer = yaml_by_id.get(upstream)
            producer_prompt = str(producer.get("prompt") or "") if producer else ""
            producer_declares_field = bool(
                producer
                and re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(str(expected_gate['artifact_field']))}(?![A-Za-z0-9_])",
                    producer_prompt,
                )
            )
            upstream_precedes = bool(
                producer and order_by_id.get(upstream, 0) < order_by_id.get(task_id, 0)
            )
            gate_matches = document_gate == yaml_gate == expected_gate
            passed = gate_matches and upstream_precedes and producer_declares_field
            rows.append(
                {
                    "task_id": task_id,
                    "gate_index": index + 1,
                    "expected_gate": deepcopy(expected_gate),
                    "document_gate": deepcopy(document_gate),
                    "yaml_gate": deepcopy(yaml_gate),
                    "gate_matches": gate_matches,
                    "upstream_precedes": upstream_precedes,
                    "producer_declares_field": producer_declares_field,
                    "broken_contract": not passed,
                    "passed": passed,
                    "terminal": True,
                }
            )
    return rows


def contract_conforms(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Require exact independent task and gate parity."""

    if len(document_rows) != 12 or len(yaml_rows) != 12 or len(gate_rows) != 13:
        return False
    sources_match = all(
        _expected_row_matches(document_rows[index], expected, index + 1)
        and _expected_row_matches(yaml_rows[index], expected, index + 1)
        for index, expected in enumerate(EXPECTED_TASKS)
    )
    return sources_match and all(row.get("passed") is True for row in gate_rows)


def _scalar(value: Any) -> Any:
    """Read repository fields that can carry a principle wrapper."""

    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _stamped_flag_kinds(payload: Mapping[str, Any]) -> list[str]:
    """List exact stored flag names without inferring a replacement name."""

    kinds = []
    for field in ("corrigendum_pending", "adversarial_flags"):
        rows = payload.get(field)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping) and row.get("kind"):
                kinds.append(str(row["kind"]))
    return list(dict.fromkeys(kinds))


def _live_flag_kinds(live_flags: Sequence[Mapping[str, Any]]) -> list[str]:
    """List fresh critical flag names that exclude current evidence."""

    return list(
        dict.fromkeys(
            str(row.get("kind"))
            for row in live_flags
            if row.get("kind") and str(row.get("severity") or "").lower() == "critical"
        )
    )


def _prefix_class(honest_verdict: str) -> str | None:
    """Read a class stated at the start of an honest verdict."""

    text = honest_verdict.strip().lower()
    if text.startswith("blocked_"):
        return "blocked"
    if text.startswith("partial_"):
        return "partial"
    for verdict_class in (
        "circular_positive",
        "disqualified",
        "positive",
        "null",
    ):
        if re.match(
            rf"^(?:complete|success|passed|shipped)[ :_]+{verdict_class}(?:[ :_]|$)",
            text,
        ):
            return verdict_class
    if re.match(r"^(?:complete|success|passed|shipped):", text):
        return "positive"
    return None


def classify_task(
    task: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    live_flags: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Assign one terminal branch while preserving absence and exclusions."""

    base: JsonDict = {
        "number": int(task["number"]),
        "task_id": str(task["task_id"]),
        "title": str(task["title"]),
        "deliverable": str(task["deliverable"]),
        "terminal": True,
        "science_task": int(task["number"]) in SCIENCE_TASKS,
        "science_positive": False,
    }
    if payload is None:
        return {
            **base,
            "artifact_present": False,
            "branch_status": "absent",
            "declared_verdict_class": None,
            "structural_verdict_class": "blocked",
            "verdict_class": "blocked",
            "honest_verdict": "blocked_expected_artifact_absent",
            "stamped_flag_kinds": [],
            "live_flag_kinds": [],
            "excluded": True,
            "row_backed": False,
        }

    honest = str(_scalar(payload.get("honest_verdict")) or "")
    declared = str(_scalar(payload.get("verdict_class")) or "")
    status = str(_scalar(payload.get("status")) or "").lower()
    pre_gate = (
        payload.get("blocked_at_layer") == "conductor_pre_gate"
        or (
            status == "blocked"
            and not isinstance(payload.get("field_principles"), Mapping)
            and isinstance(payload.get("gates_evaluated"), list)
        )
    )
    stamped_kinds = _stamped_flag_kinds(payload)
    live_kinds = _live_flag_kinds(live_flags)
    source_flagged = _scalar(payload.get("flagged_adversarial")) is True
    excluded = source_flagged or bool(live_kinds)
    prefix_class = _prefix_class(honest)
    row_backed = isinstance(payload.get("rows"), list) and bool(payload.get("rows"))

    if excluded:
        branch, final_class = "disqualified", "disqualified"
    elif pre_gate:
        branch, final_class = "pre_gate_blocked", "blocked"
    elif status == "blocked" or honest.startswith("blocked_") or declared == "blocked":
        branch, final_class = "artifact_blocked", "blocked"
    elif declared not in VERDICT_CLASSES:
        branch, final_class = "partial", "partial"
    elif prefix_class is not None and prefix_class != declared:
        branch, final_class = "disqualified", "disqualified"
    elif declared == "positive" and _scalar(payload.get("verifier_is_oracle")) is True:
        branch, final_class = "circular_positive", "circular_positive"
    else:
        branch, final_class = declared, declared
    return {
        **base,
        "artifact_present": True,
        "branch_status": branch,
        "declared_verdict_class": declared or None,
        "structural_verdict_class": prefix_class,
        "verdict_class": final_class,
        "honest_verdict": honest,
        "stamped_flag_kinds": stamped_kinds,
        "live_flag_kinds": live_kinds,
        "excluded": final_class == "disqualified",
        "row_backed": row_backed,
    }


def build_flagged_exclusion_rows(task_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return only tasks excluded by exact stamped or fresh flag names."""

    return [
        {
            "number": row["number"],
            "task_id": row["task_id"],
            "stamped_flag_kinds": list(row.get("stamped_flag_kinds") or []),
            "live_flag_kinds": list(row.get("live_flag_kinds") or []),
            "excluded": True,
        }
        for row in task_rows
        if row.get("excluded") is True
        and (row.get("stamped_flag_kinds") or row.get("live_flag_kinds"))
    ]


def recompute_runtime(payload: Mapping[str, Any] | None) -> JsonDict:
    """Reduce one completion decision from the three family rows."""

    rows = payload.get("live_generation_rows") if payload else None
    usable = [row for row in rows or [] if isinstance(row, Mapping)]
    by_model = {str(row.get("model_id")): row for row in usable}
    family_rows = []
    for model in REQUIRED_MODELS:
        row = by_model.get(model, {})
        checks = {
            "terminal": row.get("terminal_state") == "complete",
            "generated_tokens": isinstance(row.get("completion_tokens"), int)
            and row.get("completion_tokens", 0) > 0,
            "live_cuda": row.get("live_cuda") is True,
            "lease_consistent": row.get("lease_consistent", row.get("process_owned")) is True,
            "teardown_complete": row.get("teardown_complete") is True,
            "vram_released": row.get("vram_release_passed") is True,
            "owned_process": row.get("process_owned") is True,
            "owned_process_absent": row.get("owned_process_absent") is True,
        }
        family_rows.append({"model_id": model, **checks, "passed": all(checks.values())})
    completed = sum(row["passed"] for row in family_rows)
    recomputed = int(completed == len(REQUIRED_MODELS))
    reported = _scalar(payload.get("lease_aware_runtime_ready_score")) if payload else None
    return {
        "available": payload is not None,
        "family_count": len(usable),
        "completed_family_count": completed,
        "family_rows": family_rows,
        "recomputed_ready_score": recomputed,
        "reported_ready_score": reported,
        "reported_matches": reported == recomputed,
        "infrastructure_only": True,
    }


def _outcome_pass(value: Any) -> bool:
    """Recognize an explicit successful outcome without using truthy prose."""

    if value is True:
        return True
    return str(value or "").lower() in {
        "pass",
        "passed",
        "correct",
        "equivalent",
        "success",
        "valid",
    }


def recompute_constraints(
    bank: Mapping[str, Any] | None,
    certification: Mapping[str, Any] | None,
) -> JsonDict:
    """Recompute bank, exact mapping, policy, and held-out headroom counts."""

    bank_rows = [row for row in (bank or {}).get("rows", []) if isinstance(row, Mapping)]
    candidates = [
        row
        for row in (certification or {}).get("per_candidate_rows", [])
        if isinstance(row, Mapping)
    ]
    groups = [
        row
        for row in (certification or {}).get("per_group_results", [])
        if isinstance(row, Mapping)
    ]
    policy_rows = [
        row
        for row in (certification or {}).get("policy_selection_rows", [])
        if isinstance(row, Mapping)
    ]
    metric_rows = []
    for split in ("calibration", "heldout"):
        split_rows = [row for row in candidates if row.get("split") == split]
        for schedule in sorted({str(row.get("schedule_id")) for row in split_rows}):
            schedule_rows = [row for row in split_rows if str(row.get("schedule_id")) == schedule]
            metric_rows.append(
                {
                    "split": split,
                    "schedule_id": schedule,
                    "candidate_count": len(schedule_rows),
                    "parse_success_count": sum(row.get("parse_success") is True for row in schedule_rows),
                    "schema_success_count": sum(
                        _outcome_pass(row.get("schema_outcome")) for row in schedule_rows
                    ),
                    "domain_success_count": sum(
                        _outcome_pass(row.get("domain_correspondence_outcome"))
                        for row in schedule_rows
                    ),
                    "objective_success_count": sum(
                        _outcome_pass(row.get("objective_direction_outcome"))
                        and _outcome_pass(row.get("objective_order_outcome"))
                        for row in schedule_rows
                    ),
                    "exact_semantic_success_count": sum(
                        row.get("exact_semantic_success") is True for row in schedule_rows
                    ),
                }
            )
    calibration_labels = sorted(
        {
            bool(row.get("exact_semantic_success"))
            for row in candidates
            if row.get("split") == "calibration"
        }
    )
    selected = next((row for row in policy_rows if row.get("selected") is True), None)
    headroom = [row for row in groups if row.get("has_heldout_headroom") is True]
    exact_success_count = sum(row.get("exact_semantic_success") is True for row in candidates)
    result = {
        "available": bank is not None or certification is not None,
        "bank_candidate_count": len(bank_rows),
        "bank_terminal_count": sum(row.get("terminal") is True for row in bank_rows),
        "certified_candidate_count": len(candidates),
        "certified_terminal_count": sum(row.get("terminal") is True for row in candidates),
        "parse_success_count": sum(row.get("parse_success") is True for row in candidates),
        "schema_success_count": sum(_outcome_pass(row.get("schema_outcome")) for row in candidates),
        "domain_success_count": sum(
            _outcome_pass(row.get("domain_correspondence_outcome")) for row in candidates
        ),
        "objective_success_count": sum(
            _outcome_pass(row.get("objective_direction_outcome"))
            and _outcome_pass(row.get("objective_order_outcome"))
            for row in candidates
        ),
        "exact_semantic_success_count": exact_success_count,
        "authority_agreement_count": sum(row.get("authorities_agree") is True for row in candidates),
        "heldout_headroom_group_count": len(headroom),
        "heldout_headroom_total": sum(
            int(row.get("within_group_exact_headroom") or 0) for row in headroom
        ),
        "calibration_label_values": calibration_labels,
        "selected_policy": selected.get("schedule_id") if selected else None,
        "selection_used_heldout_labels": selected.get("selection_uses_heldout_labels")
        if selected
        else None,
        "metric_rows": metric_rows,
        "selection_verdict_class": "circular_positive"
        if len(headroom) >= 2 and exact_success_count > 0
        else "null",
    }
    return result


def recompute_pwa(
    pwa: Mapping[str, Any] | None,
    certification: Mapping[str, Any] | None,
) -> JsonDict:
    """Require label variation and real PWA, MILP, certificate, and test rows."""

    source_labels = {
        bool(row.get("exact_semantic_success"))
        for row in (certification or {}).get("per_candidate_rows", [])
        if isinstance(row, Mapping) and row.get("split") == "calibration"
    }
    feature_rows = [row for row in (pwa or {}).get("feature_rows", []) if isinstance(row, Mapping)]
    unit_rows = [row for row in (pwa or {}).get("pwa_unit_rows", []) if isinstance(row, Mapping)]
    milp_rows = [row for row in (pwa or {}).get("milp_query_rows", []) if isinstance(row, Mapping)]
    certificate_rows = [
        row for row in (pwa or {}).get("invariant_certificate_rows", []) if isinstance(row, Mapping)
    ]
    heldout_rows = [
        row for row in (pwa or {}).get("heldout_comparison_rows", []) if isinstance(row, Mapping)
    ]
    labels_nonconstant = source_labels == {False, True}
    ready = int(
        labels_nonconstant
        and bool(feature_rows)
        and bool(unit_rows)
        and bool(milp_rows)
        and bool(certificate_rows)
        and bool(heldout_rows)
        and all(row.get("terminal", True) is True for row in unit_rows + milp_rows + certificate_rows)
    )
    reported = _scalar(pwa.get("certified_pwa_energy_ready_score")) if pwa else None
    return {
        "available": pwa is not None,
        "feature_count": len(feature_rows),
        "calibration_label_values": sorted(source_labels),
        "calibration_labels_nonconstant": labels_nonconstant,
        "pwa_unit_count": len(unit_rows),
        "milp_query_count": len(milp_rows),
        "certificate_count": len(certificate_rows),
        "heldout_comparison_count": len(heldout_rows),
        "recomputed_ready_score": ready,
        "reported_ready_score": reported,
        "reported_matches": reported == ready,
    }


def recompute_self_learning(payload: Mapping[str, Any] | None) -> JsonDict:
    """Recompute arm successes, plasticity, forgetting, and rollback safety."""

    rows = [row for row in (payload or {}).get("rows", []) if isinstance(row, Mapping)]
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row.get("arm"))].append(row)
    success_by_arm = {
        arm: sum(row.get("exact_success") is True for row in arm_rows)
        for arm, arm_rows in sorted(by_arm.items())
    }
    event_count_by_arm = {arm: len(arm_rows) for arm, arm_rows in sorted(by_arm.items())}
    gain = success_by_arm.get("transactional_write", 0) - success_by_arm.get("read_only", 0)
    held_future = [
        row for row in (payload or {}).get("held_future_rows", []) if isinstance(row, Mapping)
    ]
    losses = [
        -float(row["paired_delta"])
        for row in held_future
        if isinstance(row.get("paired_delta"), (int, float)) and row["paired_delta"] < 0
    ]
    max_forgetting = max(losses, default=0.0)
    rollback_rows = [
        row for row in (payload or {}).get("rollback_rows", []) if isinstance(row, Mapping)
    ]
    rollback_safe = bool(rollback_rows) and all(row.get("passed") is True for row in rollback_rows)
    complete = int(
        set(by_arm) == {"frozen", "read_only", "transactional_write"}
        and len({len(value) for value in by_arm.values()}) == 1
        and all(row.get("terminal") is True for row in rows)
        and rollback_safe
    )
    positive = int(complete == 1 and gain > 0 and max_forgetting == 0)
    return {
        "available": payload is not None,
        "row_count": len(rows),
        "event_count_by_arm": event_count_by_arm,
        "success_count_by_arm": success_by_arm,
        "chronological_gain_over_readonly": gain,
        "held_future_gain": sum(float(row.get("paired_delta") or 0) for row in held_future),
        "max_forgetting": max_forgetting,
        "rollback_safe": rollback_safe,
        "recomputed_run_complete_score": complete,
        "positive_score": positive,
        "reported_gain": _scalar(payload.get("chronological_gain_over_readonly")) if payload else None,
        "reported_max_forgetting": _scalar(payload.get("max_forgetting")) if payload else None,
    }


def _binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute AUROC from pair comparisons and award half credit to ties."""

    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(positive > negative for positive in positives for negative in negatives)
    ties = sum(positive == negative for positive in positives for negative in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute average precision from descending score ranks."""

    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels, strict=True), reverse=True)
    found = 0
    precisions = []
    for rank, (_, label) in enumerate(ordered, 1):
        if label == 1:
            found += 1
            precisions.append(found / rank)
    return sum(precisions) / positives


def recompute_spilled_energy(payload: Mapping[str, Any] | None) -> JsonDict:
    """Recompute held-out signal metrics from eligible span rows."""

    rows = [
        row for row in (payload or {}).get("per_span_results", []) if isinstance(row, Mapping)
    ]
    heldout = [row for row in rows if row.get("split") == "heldout"]
    eligible = [
        row
        for row in heldout
        if row.get("eligible") is True
        and row.get("error_label") in {0, 1}
        and isinstance(row.get("signals"), Mapping)
    ]
    signal_names = sorted(
        {
            str(name)
            for row in eligible
            for name, value in row["signals"].items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    )
    metrics = {}
    for name in signal_names:
        pairs = [
            (int(row["error_label"]), float(row["signals"][name]))
            for row in eligible
            if isinstance(row["signals"].get(name), (int, float))
            and math.isfinite(float(row["signals"][name]))
        ]
        labels = [label for label, _ in pairs]
        scores = [score for _, score in pairs]
        metrics[name] = {
            "eligible_count": len(pairs),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "auroc": _binary_auc(labels, scores),
            "auprc": _average_precision(labels, scores),
        }
    spilled_auc = (metrics.get("spilled_energy") or {}).get("auroc")
    control_aucs = [
        value["auroc"]
        for name, value in metrics.items()
        if name != "spilled_energy" and value.get("auroc") is not None
    ]
    coverage = len(eligible) / len(heldout) if heldout else 0.0
    retirement = (payload or {}).get("retirement_recommendation")
    retired = isinstance(retirement, Mapping) and retirement.get("retire") is True
    positive = int(
        not retired
        and spilled_auc is not None
        and spilled_auc > 0.5
        and (not control_aucs or spilled_auc > max(control_aucs))
        and coverage >= 0.5
    )
    return {
        "available": payload is not None,
        "heldout_count": len(heldout),
        "eligible_heldout_count": len(eligible),
        "coverage": coverage,
        "signal_metrics": metrics,
        "recomputed_requalified_score": positive,
        "reported_requalified_score": _scalar(payload.get("spilled_energy_requalified_score"))
        if payload
        else None,
        "retired": retired,
        "propose_retry": retirement.get("propose_retry") if isinstance(retirement, Mapping) else None,
        "retirement_reason": retirement.get("reason") if isinstance(retirement, Mapping) else None,
    }


def recompute_arc(payload: Mapping[str, Any] | None) -> JsonDict:
    """Recompute ARC accuracy, control delta, and live-path reachability."""

    transitions = [
        row for row in (payload or {}).get("per_transition_rows", []) if isinstance(row, Mapping)
    ]
    exact_values = [
        bool(row.get("exact", row.get("exact_match", row.get("prediction_correct", False))))
        for row in transitions
    ]
    changing = [row for row in transitions if row.get("changing") is True]
    noops = [row for row in transitions if row.get("changing") is False]
    delta_rows = [
        row
        for row in (payload or {}).get("paired_control_delta_rows", [])
        if isinstance(row, Mapping)
    ]
    deltas = []
    for row in delta_rows:
        value = next(
            (
                row[key]
                for key in ("delta", "accuracy_delta", "paired_delta")
                if isinstance(row.get(key), (int, float))
            ),
            None,
        )
        if value is not None:
            deltas.append(float(value))
    live_rows = [
        row for row in (payload or {}).get("live_path_trace_rows", []) if isinstance(row, Mapping)
    ]
    reachable = bool(live_rows) and all(
        row.get("reachable", row.get("passed", False)) is True for row in live_rows
    )

    def accuracy(unit_rows: Sequence[Mapping[str, Any]]) -> float | None:
        if not unit_rows:
            return None
        return sum(
            bool(row.get("exact", row.get("exact_match", row.get("prediction_correct", False))))
            for row in unit_rows
        ) / len(unit_rows)

    return {
        "available": payload is not None,
        "transition_count": len(transitions),
        "heldout_exact_accuracy": sum(exact_values) / len(exact_values) if exact_values else None,
        "heldout_changing_accuracy": accuracy(changing),
        "heldout_noop_accuracy": accuracy(noops),
        "control_delta_count": len(deltas),
        "strongest_control_delta": max(deltas) if deltas else None,
        "live_path_trace_count": len(live_rows),
        "live_path_reachable_score": int(reachable),
        "generalization_positive_score": int(
            reachable and bool(deltas) and max(deltas) > 0 and bool(transitions)
        ),
    }


def recompute_hybrid_selection(payload: Mapping[str, Any] | None) -> JsonDict:
    """Recompute hybrid wins and retain exact-filter circularity."""

    groups = [
        row for row in (payload or {}).get("per_group_results", []) if isinstance(row, Mapping)
    ]
    if not groups:
        return {
            "available": False,
            "group_count": 0,
            "likelihood_success_count": 0,
            "hybrid_success_count": 0,
            "hybrid_gain": None,
            "exact_filter_used": False,
            "verdict_class": "blocked",
        }
    likelihood = sum(row.get("likelihood_success") is True for row in groups)
    hybrid = sum(row.get("hybrid_success") is True for row in groups)
    exact_filter = all(row.get("exact_filter_used") is True for row in groups)
    gain = hybrid - likelihood
    verdict_class = "circular_positive" if gain > 0 and exact_filter else "null"
    return {
        "available": True,
        "group_count": len(groups),
        "terminal_group_count": sum(row.get("terminal") is True for row in groups),
        "likelihood_success_count": likelihood,
        "hybrid_success_count": hybrid,
        "hybrid_gain": gain,
        "exact_filter_used": exact_filter,
        "verdict_class": verdict_class,
    }


def blocked_causes(task: Mapping[str, Any], payload: Mapping[str, Any] | None) -> list[JsonDict]:
    """Normalize every blocked cause into failed, expected, and observed fields."""

    base = {"number": int(task["number"]), "task_id": str(task["task_id"])}
    if payload is None:
        return [
            {
                **base,
                "cause_type": "absent_artifact",
                "failed_check": "artifact_presence",
                "expected_value": str(task["deliverable"]),
                "observed_value": None,
            }
        ]
    gates = payload.get("gates_evaluated")
    if isinstance(gates, list):
        failed = [row for row in gates if isinstance(row, Mapping) and row.get("passed") is False]
        if failed:
            return [
                {
                    **base,
                    "cause_type": "upstream_gate",
                    "failed_check": f"{row.get('upstream')}.{row.get('artifact_field')}",
                    "expected_value": row.get("expected", row.get("value")),
                    "observed_value": row.get("actual", row.get("observed")),
                }
                for row in failed
            ]
    summary = payload.get("gate_check_summary")
    source_rows: list[Mapping[str, Any]] = []
    if isinstance(summary, list):
        source_rows = [row for row in summary if isinstance(row, Mapping)]
    elif isinstance(summary, Mapping):
        if isinstance(summary.get("failed_checks"), list):
            source_rows = [
                row for row in summary["failed_checks"] if isinstance(row, Mapping)
            ]
        elif isinstance(summary.get("checks"), list):
            source_rows = [
                row
                for row in summary["checks"]
                if isinstance(row, Mapping) and row.get("passed") is False
            ]
        elif summary.get("failed_check") is not None:
            source_rows = [summary]
    return [
        {
            **base,
            "cause_type": "local_check",
            "failed_check": row.get("failed_check", row.get("check")),
            "expected_value": row.get("expected_value", row.get("expected")),
            "observed_value": row.get("observed_value", row.get("observed")),
        }
        for row in source_rows
    ]


def _sha256(path: Path) -> str:
    """Hash one primary source with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> tuple[JsonDict | None, str | None]:
    """Read one artifact independently and preserve its parse failure."""

    if not path.is_file():
        return None, "absent"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "JSON root is not an object"
    return value, None


def _load_module(path: Path, name: str) -> Any:
    """Load a repository checker from the selected project root."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fresh_flags(root: Path, artifact_path: Path) -> tuple[list[JsonDict], str | None]:
    """Run the current verifier when available and keep tool errors separate."""

    verifier_path = root / ADVERSARIAL_PATH
    if not verifier_path.is_file():
        return [], "adversarial verifier unavailable"
    try:
        verifier = _load_module(verifier_path, f"_carnot_adversarial_6983_{artifact_path.stem}")
        report = verifier.verify_artifact(artifact_path)
    except Exception as exc:  # noqa: BLE001 - the receipt must preserve checker failures.
        return [], f"{type(exc).__name__}: {exc}"
    flags = report.get("flags", []) if isinstance(report, Mapping) else []
    return [dict(row) for row in flags if isinstance(row, Mapping)], None


def _field_principles() -> JsonDict:
    """Give every required field one short scientific reason."""

    special = {
        "v611_capstone_complete_score": "Completion counts finished reconciliation, not milestone success.",
        "v611_task_contract_conforms_score": "Contract parity prevents omitted work from becoming silent success.",
        "v611_science_positive_score": "Only unflagged row-backed non-circular science can raise this score.",
        "flagged_exclusion_rows": "Exact flag names keep quarantined evidence outside every headline.",
        "blocked_cause_rows": "Expected and observed values make blocked branches causal and actionable.",
        "v612_handoff_rows": "The earliest open cause prevents another unchanged retry.",
        "rows": "Per-task rows let a reader recompute every milestone claim.",
        "source_artifact_hashes": "Hashes bind conclusions to exact primary bytes.",
        "reproducibility_checksum": "The digest exposes later changes to the capstone receipt.",
    }
    return {
        field: special.get(field, f"The {field} field exposes one falsifiable reconciliation fact.")
        for field in sorted(REQUIRED_ARTIFACT_FIELDS)
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable content while excluding duration and the self digest."""

    body = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    data = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _empty_artifact(run_date: str) -> JsonDict:
    """Return a complete blocked schema before contract preflight."""

    return {
        "schema": "carnot.v611_capstone.v1",
        "experiment_id": 6983,
        "run_date": run_date,
        "status": "blocked",
        "field_principles": _field_principles(),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "expected_task_ids": EXPECTED_TASK_IDS,
        "observed_task_ids": [],
        "document_contract_rows": [],
        "yaml_contract_rows": [],
        "gate_contract_rows": [],
        "rows": [],
        "per_task_results": [],
        "artifact_presence_rows": [],
        "flagged_exclusion_rows": [],
        "pre_gate_rows": [],
        "verdict_class_rows": [],
        "branch_status_rows": [],
        "runtime_recomputation": recompute_runtime(None),
        "constraint_recomputation": recompute_constraints(None, None),
        "pwa_certificate_recomputation": recompute_pwa(None, None),
        "self_learning_recomputation": recompute_self_learning(None),
        "spilled_energy_recomputation": recompute_spilled_energy(None),
        "arc_recomputation": recompute_arc(None),
        "hybrid_selection_recomputation": recompute_hybrid_selection(None),
        "prior_failure_disposition_rows": [],
        "blocked_cause_rows": [],
        "command_receipt_rows": [],
        "v612_handoff_rows": [],
        "v611_capstone_complete_score": 0,
        "v611_task_contract_conforms_score": 0,
        "v611_science_positive_score": 0,
        "paper_ready": False,
        "paper_ready_source": {"path": str(PUBLICATION_GATE_PATH), "available": False},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }


def _run_command(root: Path, name: str, argv: Sequence[str]) -> JsonDict:
    """Run one audit and preserve its exact terminal receipt."""

    command = shlex.join(str(value) for value in argv)
    try:
        result = subprocess.run(
            list(argv),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "exit_code": 124,
            "stdout": str(exc.output or ""),
            "stderr": str(exc.stderr or ""),
            "outcome": "tool_failure",
            "terminal": False,
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
        }
    return {
        "name": name,
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "outcome": "pass" if result.returncode == 0 else "finding",
        "terminal": True,
    }


def _publication_ready(root: Path) -> tuple[bool, JsonDict, JsonDict | None]:
    """Read the stable G1-G4 gate instead of inventing readiness."""

    script = root / PUBLICATION_GATE_PATH
    if not script.is_file():
        return False, {"path": str(PUBLICATION_GATE_PATH), "available": False}, None
    receipt = _run_command(
        root,
        "publication_gate",
        [str(root / ".venv/bin/python"), str(PUBLICATION_GATE_PATH), "--json"],
    )
    try:
        payload = json.loads(receipt["stdout"]) if receipt["exit_code"] == 0 else {}
    except json.JSONDecodeError:
        payload = {}
    source = {
        "path": str(PUBLICATION_GATE_PATH),
        "available": True,
        "exit_code": receipt["exit_code"],
        "gates": payload.get("gates"),
        "unmet_gates": payload.get("unmet_gates"),
    }
    return payload.get("paper_ready") is True, source, receipt


def run_operational_commands(root: Path, artifact_paths: Sequence[Path]) -> list[JsonDict]:
    """Run bounded roadmap, row, ARC, and clutter audits for the receipt."""

    python = root / ".venv/bin/python"
    schema_code = (
        "from pathlib import Path; import sys,yaml; sys.path.insert(0,'scripts'); "
        "from roadmap_schema import Roadmap; "
        "Roadmap.model_validate(yaml.safe_load(Path('research-roadmap.yaml').read_text())); "
        "print('roadmap schema validation clean')"
    )
    commands = [
        ("roadmap_schema", [str(python), "-c", schema_code]),
        (
            "roadmap_gate_audit",
            [str(python), "scripts/audit_roadmap_gates.py", str(ROADMAP_PATH)],
        ),
        (
            "exclusion_manifest_lint",
            [str(python), "scripts/exclusion_manifest_lint.py", str(ROADMAP_PATH)],
        ),
        (
            "verdict_row_consistency_lint",
            [str(python), str(ROW_LINT_PATH), *[str(path) for path in artifact_paths]],
        ),
        (
            "arc_live_path_lint",
            [str(python), "scripts/arc_llm_on_liveness_lint.py", "--self-test"],
        ),
        (
            "root_clutter_check",
            [str(python), "scripts/root_clutter_sweep.py"],
        ),
    ]
    return [_run_command(root, name, argv) for name, argv in commands]


def _source_hashes(root: Path, artifact_paths: Sequence[Path]) -> JsonDict:
    """Hash every available contract, checker, and upstream artifact."""

    paths = [
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        SPEC_PATH,
        DESIGN_PATH,
        ROADMAP_PATH,
        EXCLUSION_PATH,
        ADVERSARIAL_PATH,
        ROW_LINT_PATH,
        PUBLICATION_GATE_PATH,
        Path("python/carnot/experiment_6983_v611_capstone.py"),
        Path("scripts/experiments/experiment_6983_v611_capstone.py"),
        Path("tests/python/test_experiment_6983_v611_capstone.py"),
        *artifact_paths,
    ]
    return {
        path.as_posix(): _sha256(root / path)
        for path in paths
        if (root / path).is_file()
    }


def _prior_failure_rows(
    yaml_rows: Sequence[Mapping[str, Any]],
    task_rows: Mapping[int, Mapping[str, Any]],
    payloads: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    """Preserve prior outcomes and the final spilled-energy retirement."""

    rows = []
    for task in yaml_rows:
        number = int(task.get("number") or 0)
        current = task_rows.get(number, {})
        for prior in task.get("prior_failures", []):
            if not isinstance(prior, Mapping):
                continue
            row = {
                "task_id": task.get("task_id"),
                "prior_experiment_id": prior.get("experiment_id"),
                "prior_verdict": prior.get("verdict"),
                "addressed_by": prior.get("addressed_by"),
                "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                "current_branch_status": current.get("branch_status"),
                "final_retirement": False,
            }
            if number == 6980:
                retirement = payloads.get(number, {}).get("retirement_recommendation")
                row["final_retirement"] = bool(
                    isinstance(retirement, Mapping) and retirement.get("retire") is True
                )
                row["retirement_reason"] = (
                    retirement.get("reason") if isinstance(retirement, Mapping) else None
                )
            rows.append(row)
    return rows


def _v612_handoff(pwa: Mapping[str, Any], arc: Mapping[str, Any]) -> list[JsonDict]:
    """Recommend the earliest changed-state action and exclude closed retries."""

    if pwa.get("calibration_labels_nonconstant") is not True:
        boundary_task = "exp6977-certified-pwa-kan-energy"
        cause = "The calibration candidates contain one exact label, so no residual can learn an error boundary."
        action = (
            "Create a prospectively frozen calibration candidate set with both exact successes and failures. "
            "Keep held-out labels sealed, then train and certify PWA energy once."
        )
    elif arc.get("available") and arc.get("live_path_reachable_score") != 1:
        boundary_task = "exp6981-arc-live-engine-generalization-audit"
        cause = "No immutable completed live run binds the engine to transitions and the scored policy."
        action = "Capture one complete live run-engine binding before another ARC generalization audit."
    else:
        boundary_task = "exp6978-transactional-constraint-self-learning"
        cause = "Transactional memory did not improve exact success over read-only memory."
        action = "Change the learned policy signal before another matched-arm chronological run."
    return [
        {
            "rank": 1,
            "boundary_task_id": boundary_task,
            "unresolved_cause": cause,
            "recommended_action": action,
            "requires_changed_state": True,
            "forbidden_unchanged_retries": [
                "schema_only_retry",
                "finite_id_transport",
                "headroom_free_selection",
                "exact_slot_requalification",
                "token_budget_only_arc_induction",
                "public_game_resolve",
                "unchanged_hardware_task",
            ],
            "retired_upstream_dependency_added": False,
        }
    ]


def build_artifact(root: Path, run_date: str, *, run_commands: bool = True) -> JsonDict:
    """Build the terminal V611 capstone from independent primary evidence."""

    started = time.monotonic()
    artifact = _empty_artifact(run_date)
    contract_paths = {
        "active_roadmap": ROADMAP_PATH,
        "design_document": DESIGN_PATH,
    }
    preconditions = [
        {
            "resource": name,
            "path": path.as_posix(),
            "available": (root / path).is_file() and (root / path).stat().st_size > 0,
        }
        for name, path in contract_paths.items()
    ]
    artifact["preconditions_checked"] = preconditions
    if not all(row["available"] for row in preconditions):
        artifact["gate_check_summary"] = {
            "failed_check": "contract_preconditions",
            "expected_value": "both contract files present",
            "observed_value": [row["path"] for row in preconditions if not row["available"]],
            "passed": False,
        }
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    design_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
    try:
        roadmap = load_yaml(root / ROADMAP_PATH)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError):
        roadmap = {}
    document_rows = parse_document_contract(design_text)
    yaml_rows = parse_yaml_contract(roadmap)
    gate_rows = build_gate_contract_rows(document_rows, yaml_rows)
    conforms = contract_conforms(document_rows, yaml_rows, gate_rows)
    yaml_by_number = {
        int(row["number"]): row for row in yaml_rows if isinstance(row.get("number"), int)
    }

    payloads: dict[int, JsonDict] = {}
    task_rows: dict[int, JsonDict] = {}
    presence_rows = []
    artifact_paths = []
    live_audit_rows = []
    for expected in EXPECTED_TASKS[:-1]:
        number = int(expected["number"])
        path = root / str(expected["deliverable"])
        artifact_paths.append(Path(str(expected["deliverable"])))
        payload, read_error = _load_json(path)
        if payload is not None:
            payloads[number] = payload
        flags, verifier_error = _fresh_flags(root, path) if payload is not None else ([], None)
        task = yaml_by_number.get(number, expected)
        task_row = classify_task(task, payload, flags)
        if read_error not in {None, "absent"}:
            task_row.update(
                {
                    "branch_status": "partial",
                    "verdict_class": "partial",
                    "honest_verdict": "partial_unreadable_artifact",
                    "excluded": True,
                    "read_error": read_error,
                }
            )
        task_rows[number] = task_row
        presence_rows.append(
            {
                "number": number,
                "task_id": expected["task_id"],
                "path": expected["deliverable"],
                "present": payload is not None,
                "read_error": read_error,
                "attempted_independently": True,
            }
        )
        live_audit_rows.append(
            {
                "number": number,
                "path": expected["deliverable"],
                "flag_kinds": [str(row.get("kind")) for row in flags if row.get("kind")],
                "verifier_error": verifier_error,
            }
        )

    runtime = recompute_runtime(payloads.get(6973))
    constraints = recompute_constraints(payloads.get(6975), payloads.get(6976))
    pwa = recompute_pwa(payloads.get(6977), payloads.get(6976))
    learning = recompute_self_learning(payloads.get(6978))
    spilled = recompute_spilled_energy(payloads.get(6980))
    arc = recompute_arc(payloads.get(6981))
    hybrid = recompute_hybrid_selection(payloads.get(6982))

    science_positive_by_number = {
        6976: False,
        6977: pwa.get("recomputed_ready_score") == 1,
        6978: learning.get("positive_score") == 1,
        6980: spilled.get("recomputed_requalified_score") == 1,
        6981: arc.get("generalization_positive_score") == 1,
        6982: False,
    }
    for number, value in science_positive_by_number.items():
        row = task_rows.get(number)
        if row is not None:
            row["science_positive"] = bool(
                value
                and row["verdict_class"] == "positive"
                and row.get("excluded") is not True
                and row.get("row_backed") is True
            )

    science_score = int(any(row.get("science_positive") is True for row in task_rows.values()))
    verdict_class = "positive" if science_score else "null"
    honest_verdict = POSITIVE_VERDICT if science_score else NULL_VERDICT
    current = {
        "number": 6983,
        "task_id": EXPECTED_TASKS[-1]["task_id"],
        "title": EXPECTED_TASKS[-1]["title"],
        "deliverable": EXPECTED_TASKS[-1]["deliverable"],
        "terminal": True,
        "science_task": False,
        "science_positive": False,
        "artifact_present": True,
        "branch_status": verdict_class,
        "declared_verdict_class": verdict_class,
        "structural_verdict_class": verdict_class,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "stamped_flag_kinds": [],
        "live_flag_kinds": [],
        "excluded": False,
        "row_backed": True,
    }
    task_rows[6983] = current
    presence_rows.append(
        {
            "number": 6983,
            "task_id": EXPECTED_TASKS[-1]["task_id"],
            "path": EXPECTED_TASKS[-1]["deliverable"],
            "present": True,
            "read_error": None,
            "attempted_independently": True,
            "state": "current_synthesis",
        }
    )
    ordered_rows = [task_rows[number] for number in range(6972, 6984)]
    blocked_rows = []
    for expected in EXPECTED_TASKS[:-1]:
        number = int(expected["number"])
        if task_rows[number]["branch_status"] in {
            "absent",
            "pre_gate_blocked",
            "artifact_blocked",
        }:
            blocked_rows.extend(blocked_causes(expected, payloads.get(number)))
    blocked_rows.extend(
        {
            "number": _number(row.get("task_id")),
            "task_id": row.get("task_id"),
            "cause_type": "broken_contract",
            "failed_check": f"gate_{row.get('gate_index')}",
            "expected_value": row.get("expected_gate"),
            "observed_value": row.get("yaml_gate"),
        }
        for row in gate_rows
        if row.get("broken_contract") is True
    )
    paper_ready, paper_source, publication_receipt = _publication_ready(root)
    receipts = run_operational_commands(root, artifact_paths) if run_commands else []
    if publication_receipt is not None:
        receipts.append(publication_receipt)
    receipts.append(
        {
            "name": "independent_artifact_reads",
            "command": "internal read of each expected V611 artifact path",
            "exit_code": 0,
            "stdout": json.dumps(presence_rows, sort_keys=True),
            "stderr": "",
            "outcome": "pass",
            "terminal": True,
        }
    )
    receipts.append(
        {
            "name": "fresh_adversarial_rechecks",
            "command": "internal current verifier pass over each present V611 artifact",
            "exit_code": 0,
            "stdout": json.dumps(live_audit_rows, sort_keys=True),
            "stderr": "",
            "outcome": "pass",
            "terminal": True,
        }
    )
    complete = int(
        len(ordered_rows) == 12
        and all(row.get("terminal") is True for row in ordered_rows)
        and len(presence_rows) == 12
        and len(document_rows) == 12
        and len(yaml_rows) == 12
    )
    prior_rows = _prior_failure_rows(yaml_rows, task_rows, payloads)
    handoff = _v612_handoff(pwa, arc)
    artifact.update(
        {
            "status": "complete",
            "source_artifact_hashes": _source_hashes(root, artifact_paths),
            "observed_task_ids": [str(row.get("task_id")) for row in yaml_rows],
            "document_contract_rows": [
                {
                    **dict(row),
                    "passed": index < len(EXPECTED_TASKS)
                    and _expected_row_matches(row, EXPECTED_TASKS[index], index + 1),
                }
                for index, row in enumerate(document_rows)
            ],
            "yaml_contract_rows": [
                {
                    **dict(row),
                    "passed": index < len(EXPECTED_TASKS)
                    and _expected_row_matches(row, EXPECTED_TASKS[index], index + 1),
                }
                for index, row in enumerate(yaml_rows)
            ],
            "gate_contract_rows": gate_rows,
            "rows": ordered_rows,
            "per_task_results": ordered_rows,
            "artifact_presence_rows": presence_rows,
            "flagged_exclusion_rows": build_flagged_exclusion_rows(ordered_rows),
            "pre_gate_rows": [
                row for row in ordered_rows if row["branch_status"] == "pre_gate_blocked"
            ],
            "verdict_class_rows": [
                {
                    "number": row["number"],
                    "task_id": row["task_id"],
                    "declared": row["declared_verdict_class"],
                    "structural": row["structural_verdict_class"],
                    "final": row["verdict_class"],
                }
                for row in ordered_rows
            ],
            "branch_status_rows": ordered_rows,
            "runtime_recomputation": runtime,
            "constraint_recomputation": constraints,
            "pwa_certificate_recomputation": pwa,
            "self_learning_recomputation": learning,
            "spilled_energy_recomputation": spilled,
            "arc_recomputation": arc,
            "hybrid_selection_recomputation": hybrid,
            "prior_failure_disposition_rows": prior_rows,
            "blocked_cause_rows": blocked_rows,
            "command_receipt_rows": receipts,
            "v612_handoff_rows": handoff,
            "v611_capstone_complete_score": complete,
            "v611_task_contract_conforms_score": int(conforms),
            "v611_science_positive_score": science_score,
            "paper_ready": paper_ready,
            "paper_ready_source": paper_source,
            "gate_check_summary": {
                "failed_check": None if science_score else "non_circular_science_positive",
                "expected_value": 1,
                "observed_value": science_score,
                "passed": bool(science_score),
                "contract_conforms": conforms,
                "terminal_task_count": len(ordered_rows),
            },
            "verdict_class": verdict_class,
            "honest_verdict": honest_verdict,
        }
    )
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, row scores, verdicts, and checksum."""

    errors = []
    for field in sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact)):
        errors.append(f"missing required field: {field}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be a mapping")
    else:
        for field in sorted(REQUIRED_ARTIFACT_FIELDS - set(principles)):
            errors.append(f"field_principles missing {field}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in VERDICT_CLASSES:
        errors.append("verdict_class is not closed")
    honest = str(artifact.get("honest_verdict") or "")
    expected_prefixes = {
        "positive": ("complete_positive_",),
        "circular_positive": ("complete_circular_positive_",),
        "null": ("complete_null_",),
        "blocked": ("blocked_",),
        "disqualified": ("complete_disqualified_", "disqualified_"),
        "partial": ("partial_",),
    }
    if verdict_class in expected_prefixes and not honest.startswith(expected_prefixes[verdict_class]):
        errors.append("honest_verdict prefix conflicts with verdict_class")
    branch_rows = artifact.get("branch_status_rows")
    branches_valid = bool(
        isinstance(branch_rows, list)
        and len(branch_rows) == 12
        and all(
            isinstance(row, Mapping)
            and row.get("terminal") is True
            and row.get("branch_status") in BRANCH_STATES
            for row in branch_rows
        )
    )
    contracts_present = bool(
        isinstance(artifact.get("document_contract_rows"), list)
        and len(artifact["document_contract_rows"]) == 12
        and isinstance(artifact.get("yaml_contract_rows"), list)
        and len(artifact["yaml_contract_rows"]) == 12
    )
    recomputed_complete = int(branches_valid and contracts_present)
    if artifact.get("verdict_class") == "blocked":
        recomputed_complete = 0
    if artifact.get("v611_capstone_complete_score") != recomputed_complete:
        errors.append("v611_capstone_complete_score does not match terminal rows")
    contract_score = int(
        contracts_present
        and all(row.get("passed") is True for row in artifact["document_contract_rows"])
        and all(row.get("passed") is True for row in artifact["yaml_contract_rows"])
        and isinstance(artifact.get("gate_contract_rows"), list)
        and len(artifact["gate_contract_rows"]) == 13
        and all(row.get("passed") is True for row in artifact["gate_contract_rows"])
    )
    if artifact.get("v611_task_contract_conforms_score") != contract_score:
        errors.append("v611_task_contract_conforms_score does not match contract rows")
    science_score = int(
        isinstance(branch_rows, list)
        and any(
            isinstance(row, Mapping)
            and row.get("science_positive") is True
            and row.get("excluded") is not True
            and row.get("verdict_class") == "positive"
            and row.get("row_backed") is True
            for row in branch_rows
        )
    )
    if artifact.get("v611_science_positive_score") != science_score:
        errors.append("v611_science_positive_score does not match eligible rows")
    if verdict_class == "blocked":
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, Mapping) or not all(
            key in summary for key in ("failed_check", "expected_value", "observed_value")
        ):
            errors.append("blocked verdict requires failed, expected, and observed values")
    if artifact.get("expected_task_ids") != EXPECTED_TASK_IDS:
        errors.append("expected_task_ids mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def _valid_date(value: str) -> bool:
    """Accept only the conductor's compact calendar date."""

    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return len(value) == 8 and value.isdigit()


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the V611 capstone at an explicit output path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260904")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-commands", action="store_true")
    args = parser.parse_args(argv)
    if not _valid_date(args.date):
        print("--date must use YYYYMMDD", file=sys.stderr)
        return 2
    output = args.output if args.output.is_absolute() else args.repo_root / args.output
    if args.validate:
        payload, error = _load_json(output)
        errors = [error] if error else validate_artifact(payload or {})
        if errors:
            print("\n".join(str(error) for error in errors))
            return 1
        return 0
    artifact = build_artifact(args.repo_root, args.date, run_commands=not args.no_commands)
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper owns execution.
    raise SystemExit(main())
