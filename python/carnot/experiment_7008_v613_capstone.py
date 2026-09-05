"""Build the V613 independent evidence capstone for REQ-REPORT-7008.

The reducer reads contracts and result artifacts as evidence. It does not fit
models or repair missing experiments. This keeps a completed milestone record
separate from scientific success.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_7008_v613_capstone.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
V612_CAPSTONE_PATH = Path("results/experiment_6995_v612_capstone.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
MILESTONE = "2026.09.613"
INFERENCE_SUBSTRATE = "deterministic_milestone_v613_evidence_synthesis_no_llm"
RANDOM_SEED = 7008
BLOCKED_VERDICT = "blocked_v613_capstone"
NULL_VERDICT = "complete_null_v613_capstone_no_oracle_distinct_science_positive"
POSITIVE_VERDICT = "complete_positive_v613_capstone_oracle_distinct_selection_confirmed"
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}


def _gate(upstream: str, artifact_field: str) -> JsonDict:
    """Return the one equality-gate shape used by this roadmap."""

    return {"upstream": upstream, "artifact_field": artifact_field, "op": "==", "value": 1}


EXPECTED_TASKS: tuple[JsonDict, ...] = (
    {
        "number": 6996,
        "task_id": "exp6996-v613-source-contract-preflight",
        "title": "V613 source delta and task-contract preflight",
        "deliverable": "results/experiment_6996_v613_source_contract_preflight.json",
        "gates": [],
    },
    {
        "number": 6997,
        "task_id": "exp6997-authority-sidecar-rebuild",
        "title": "Authority-only mutation sidecar and blinded learner view",
        "deliverable": "results/experiment_6997_authority_sidecar_rebuild.json",
        "gates": [],
    },
    {
        "number": 6998,
        "task_id": "exp6998-three-family-commitment-controls",
        "title": "Three-family self-commitment shortcut controls",
        "deliverable": "results/experiment_6998_three_family_commitment_controls.json",
        "gates": [_gate("exp6997-authority-sidecar-rebuild", "blinded_learner_view_ready_score")],
    },
    {
        "number": 6999,
        "task_id": "exp6999-blinded-feature-cold-audit",
        "title": "Independent blinded feature isolation and shortcut audit",
        "deliverable": "results/experiment_6999_blinded_feature_cold_audit.json",
        "gates": [
            _gate("exp6997-authority-sidecar-rebuild", "blinded_learner_view_ready_score"),
            _gate("exp6998-three-family-commitment-controls", "commitment_control_complete_score"),
        ],
    },
    {
        "number": 7000,
        "task_id": "exp7000-certified-blinded-pwa-kan",
        "title": "Certified blinded PWA-KAN constraint ranker",
        "deliverable": "results/experiment_7000_certified_blinded_pwa_kan.json",
        "gates": [_gate("exp6999-blinded-feature-cold-audit", "blinded_feature_bank_ready_score")],
    },
    {
        "number": 7001,
        "task_id": "exp7001-pwa-certificate-cold-audit",
        "title": "Fresh-process blinded PWA-KAN certificate audit",
        "deliverable": "results/experiment_7001_pwa_certificate_cold_audit.json",
        "gates": [
            _gate("exp7000-certified-blinded-pwa-kan", "pwa_candidate_artifact_complete_score")
        ],
    },
    {
        "number": 7002,
        "task_id": "exp7002-oracle-distinct-selection",
        "title": "Oracle-distinct blinded constraint selection comparison",
        "deliverable": "results/experiment_7002_oracle_distinct_selection.json",
        "gates": [
            _gate("exp7000-certified-blinded-pwa-kan", "pwa_model_ready_score"),
            _gate("exp7001-pwa-certificate-cold-audit", "pwa_certificate_confirmed_score"),
        ],
    },
    {
        "number": 7003,
        "task_id": "exp7003-per-knot-continuous-learning",
        "title": "Verifier-grounded blinded per-knot continuous self-learning",
        "deliverable": "results/experiment_7003_per_knot_continuous_learning.json",
        "gates": [
            _gate("exp7002-oracle-distinct-selection", "selection_comparison_complete_score")
        ],
    },
    {
        "number": 7004,
        "task_id": "exp7004-self-learning-cold-audit",
        "title": "Fresh-process self-learning retention and support audit",
        "deliverable": "results/experiment_7004_self_learning_cold_audit.json",
        "gates": [
            _gate("exp7003-per-knot-continuous-learning", "self_learning_run_complete_score")
        ],
    },
    {
        "number": 7005,
        "task_id": "exp7005-arc-live-envelope-audit",
        "title": "Prospective ARC live-envelope held-out engine audit",
        "deliverable": "results/experiment_7005_arc_live_envelope_audit.json",
        "gates": [],
    },
    {
        "number": 7006,
        "task_id": "exp7006-z1t-sparse-placement",
        "title": "Z1T sparse scorer placement receipt",
        "deliverable": "results/experiment_7006_z1t_sparse_placement.json",
        "gates": [
            _gate("exp7000-certified-blinded-pwa-kan", "pwa_candidate_artifact_complete_score")
        ],
    },
    {
        "number": 7007,
        "task_id": "exp7007-z1t-placement-cold-audit",
        "title": "Z1T substrate partition recomputation",
        "deliverable": "results/experiment_7007_z1t_placement_cold_audit.json",
        "gates": [_gate("exp7006-z1t-sparse-placement", "z1t_placement_receipt_complete_score")],
    },
    {
        "number": 7008,
        "task_id": "exp7008-v613-capstone",
        "title": "V613 independent evidence capstone and V614 handoff",
        "deliverable": "results/experiment_7008_v613_capstone.json",
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
    "rows",
    "expected_task_rows",
    "observed_task_rows",
    "missing_task_rows",
    "task_contract_rows",
    "gate_contract_rows",
    "gate_producer_rows",
    "prior_failure_rows",
    "verdict_class_rows",
    "blocked_diagnostic_rows",
    "flagged_artifact_rows",
    "per_branch_results",
    "evidence_isolation_rows",
    "commitment_control_rows",
    "pwa_energy_rows",
    "milp_certificate_rows",
    "selection_evidence_rows",
    "continuous_learning_rows",
    "support_and_forgetting_rows",
    "arc_live_envelope_rows",
    "arc_quality_rows",
    "z1t_typed_graph_rows",
    "z1t_placement_rows",
    "circularity_rows",
    "claim_boundary_rows",
    "source_disagreement_rows",
    "retirement_rows",
    "publication_gate_rows",
    "g1",
    "g2",
    "g3",
    "g4",
    "paper_ready",
    "unmet_gates",
    "command_receipt_rows",
    "v614_handoff_rows",
    "v613_capstone_complete_score",
    "v613_task_contract_conforms_score",
    "v613_science_positive_score",
    "solve_claimed",
    "level_claimed",
    "registry_updated",
    "hardware_executed",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}

COMMAND_NAMES = (
    "publication_gate",
    "artifact_validation",
    "adversarial_verification",
    "row_consistency_lint",
    "recurring_blocker_ledger",
    "openspec_coverage",
    "root_clutter_check",
)


def spec_anchors(text: str) -> list[str]:
    """Return scenario anchors found in specification text."""

    return re.findall(r"^####\s+(SCENARIO-[A-Z0-9-]+):", text, re.MULTILINE)


def load_yaml(path: Path) -> JsonDict:
    """Read one YAML mapping without borrowing Markdown defaults."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return payload


def _number(value: Any) -> int | None:
    """Extract an experiment number from a task identifier."""

    match = re.search(r"exp(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def _normalize_gate(raw: Mapping[str, Any]) -> JsonDict:
    """Normalize a gate while preserving its named producer field."""

    return {
        "upstream": str(raw.get("upstream") or ""),
        "artifact_field": str(raw.get("artifact_field") or ""),
        "op": str(raw.get("op") or "=="),
        "value": raw.get("value"),
    }


def _parse_document_gates(value: str) -> list[JsonDict]:
    """Parse the narrow gate notation used by the exact contract table."""

    if value.strip().lower() == "none":
        return []
    pattern = re.compile(r"`([^`]+)\.([A-Za-z0-9_]+)\s*(==|!=|>=|<=|>|<)\s*([^`]+)`")
    return [
        {
            "upstream": upstream.strip(),
            "artifact_field": field.strip(),
            "op": op,
            "value": yaml.safe_load(expected.strip()),
        }
        for upstream, field, op, expected in pattern.findall(value)
    ]


def parse_document_contract(text: str) -> list[JsonDict]:
    """Parse the Markdown contract without reading the YAML task list."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`([^`]+)`", text)
    milestone = milestone_match.group(1) if milestone_match else ""
    marker = "## Exact Task Contract"
    if marker not in text:
        return []
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*`(exp\d+[a-z0-9-]*)`\s*\|\s*([^|]+?)\s*"
        r"\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|$",
        re.MULTILINE,
    )
    return [
        {
            "source": "document",
            "order": int(order),
            "number": _number(task_id),
            "task_id": task_id.strip(),
            "title": title.strip(),
            "deliverable": deliverable.strip(),
            "milestone": milestone,
            "gates": _parse_document_gates(gates),
        }
        for order, task_id, title, deliverable, gates in pattern.findall(section)
    ]


def parse_yaml_contract(roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Parse YAML tasks independently and retain producer and failure text."""

    raw_tasks = roadmap.get("tasks")
    if not isinstance(raw_tasks, list):
        return []
    milestone = str(roadmap.get("milestone") or "")
    rows: list[JsonDict] = []
    for order, raw in enumerate(raw_tasks, 1):
        task = raw if isinstance(raw, Mapping) else {}
        rows.append(
            {
                "source": "yaml",
                "order": order,
                "number": _number(task.get("id")),
                "task_id": str(task.get("id") or ""),
                "title": str(task.get("title") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "milestone": str(task.get("milestone") or milestone),
                "gates": [
                    _normalize_gate(gate)
                    for gate in task.get("gated_on", [])
                    if isinstance(gate, Mapping)
                ],
                "prompt": str(task.get("prompt") or ""),
                "prior_failures": deepcopy(task.get("prior_failures") or []),
            }
        )
    return rows


def build_task_contract_rows(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Compare task identity, position, title, deliverable, and milestone."""

    rows: list[JsonDict] = []
    for index in range(max(len(document_rows), len(yaml_rows), len(EXPECTED_TASKS))):
        document = document_rows[index] if index < len(document_rows) else {}
        roadmap = yaml_rows[index] if index < len(yaml_rows) else {}
        expected = EXPECTED_TASKS[index] if index < len(EXPECTED_TASKS) else {}
        checks = {
            "order": document.get("order") == roadmap.get("order") == index + 1,
            "task_id": document.get("task_id") == roadmap.get("task_id") == expected.get("task_id"),
            "title": document.get("title") == roadmap.get("title") == expected.get("title"),
            "deliverable": document.get("deliverable")
            == roadmap.get("deliverable")
            == expected.get("deliverable"),
            "milestone": document.get("milestone") == roadmap.get("milestone") == MILESTONE,
        }
        failed = [name for name, passed in checks.items() if not passed]
        check = "all" if not failed else ("order" if "task_id" in failed else failed[0])
        rows.append(
            {
                "order": index + 1,
                "task_id": document.get("task_id") or roadmap.get("task_id"),
                "checks": checks,
                "check": check,
                "passed": not failed,
                "terminal": True,
            }
        )
    return rows


def build_gate_contract_rows(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Compare each task's complete ordered gate list with the fixed contract."""

    document = {str(row.get("task_id")): row for row in document_rows}
    roadmap = {str(row.get("task_id")): row for row in yaml_rows}
    rows = []
    for task in EXPECTED_TASKS:
        task_id = str(task["task_id"])
        expected = deepcopy(task["gates"])
        document_gates = deepcopy(document.get(task_id, {}).get("gates", []))
        yaml_gates = deepcopy(roadmap.get(task_id, {}).get("gates", []))
        rows.append(
            {
                "number": task["number"],
                "task_id": task_id,
                "expected": expected,
                "document": document_gates,
                "yaml": yaml_gates,
                "passed": document_gates == yaml_gates == expected,
                "terminal": True,
            }
        )
    return rows


def build_gate_producer_rows(yaml_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Confirm every gate points backward to a producer that names its field."""

    by_id = {str(row.get("task_id")): row for row in yaml_rows}
    rows = []
    for consumer in yaml_rows:
        for gate in consumer.get("gates", []):
            upstream = str(gate.get("upstream") or "")
            field = str(gate.get("artifact_field") or "")
            producer = by_id.get(upstream)
            producer_present = producer is not None
            precedes = bool(
                producer_present and int(producer.get("order", 0)) < int(consumer.get("order", 0))
            )
            same_milestone = bool(
                producer_present
                and producer.get("milestone") == consumer.get("milestone") == MILESTONE
            )
            bare = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", field))
            declared = bool(producer_present and field in str(producer.get("prompt") or ""))
            passed = producer_present and precedes and same_milestone and bare and declared
            rows.append(
                {
                    "number": consumer.get("number"),
                    "consumer": consumer.get("task_id"),
                    "upstream": upstream,
                    "artifact_field": field,
                    "producer_present": producer_present,
                    "producer_precedes_consumer": precedes,
                    "same_milestone": same_milestone,
                    "bare_field": bare,
                    "field_declared": declared,
                    "passed": passed,
                    "terminal": True,
                }
            )
    return rows


def build_prior_failure_rows(yaml_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Validate each complete prior-failure contract or explicit empty list."""

    rows = []
    for task in yaml_rows:
        priors = task.get("prior_failures", [])
        if not priors:
            rows.append(
                {
                    "number": task.get("number"),
                    "task_id": task.get("task_id"),
                    "prior_failure_present": False,
                    "passed": True,
                    "terminal": True,
                }
            )
            continue
        for raw in priors:
            prior = raw if isinstance(raw, Mapping) else {}
            checks = {
                "experiment_id": bool(str(prior.get("experiment_id") or "").strip()),
                "verdict": bool(str(prior.get("verdict") or "").strip()),
                "addressed_by": bool(str(prior.get("addressed_by") or "").strip()),
                "retire_if_same_verdict": isinstance(prior.get("retire_if_same_verdict"), bool),
            }
            rows.append(
                {
                    "number": task.get("number"),
                    "task_id": task.get("task_id"),
                    "prior_failure_present": True,
                    "experiment_id": prior.get("experiment_id"),
                    "verdict": prior.get("verdict"),
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "checks": checks,
                    "passed": all(checks.values()),
                    "terminal": True,
                }
            )
    return rows


def contract_conforms(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    producer_rows: Sequence[Mapping[str, Any]],
    prior_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Require the complete fixed contract and all producer and failure checks."""

    return bool(
        len(document_rows) == len(yaml_rows) == len(task_rows) == len(gate_rows) == 13
        and len(producer_rows) == 11
        and len(prior_rows) == 22
        and [row.get("task_id") for row in document_rows] == EXPECTED_TASK_IDS
        and [row.get("task_id") for row in yaml_rows] == EXPECTED_TASK_IDS
        and all(row.get("passed") is True for row in task_rows)
        and all(row.get("passed") is True for row in gate_rows)
        and all(row.get("passed") is True for row in producer_rows)
        and all(row.get("passed") is True for row in prior_rows)
    )


def _flag_kinds(payload: Mapping[str, Any]) -> list[str]:
    """Read explicit artifact flags without inventing concerns from prose."""

    kinds: list[str] = []
    for raw in payload.get("adversarial_flags", []) or []:
        value = raw.get("kind") if isinstance(raw, Mapping) else raw
        if value:
            kinds.append(str(value))
    if payload.get("flagged_adversarial") and not kinds:
        kinds.append("flagged_adversarial")
    return sorted(set(kinds))


def classify_task(task: Mapping[str, Any], payload: Mapping[str, Any] | None) -> JsonDict:
    """Preserve declared verdicts while treating absent inputs as terminal blocks."""

    base = {
        "number": int(task["number"]),
        "task_id": str(task["task_id"]),
        "title": str(task["title"]),
        "deliverable": str(task["deliverable"]),
        "terminal": True,
        "science_positive": False,
    }
    if payload is None:
        return {
            **base,
            "artifact_present": False,
            "declared_verdict_class": None,
            "verdict_class": "blocked",
            "honest_verdict": "blocked_expected_artifact_absent",
            "outcome": "missing",
            "flag_kinds": [],
            "eligible_for_positive": False,
        }
    declared_value = payload.get("verdict_class")
    declared = str(declared_value) if declared_value in VERDICT_CLASSES else None
    honest = str(payload.get("honest_verdict") or "")
    status = str(payload.get("status") or "").lower()
    flags = _flag_kinds(payload)
    blocked = (
        status == "blocked"
        or honest.startswith("blocked_")
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
    )
    if blocked:
        verdict_class, outcome = "blocked", "blocked"
    elif status == "rejected":
        verdict_class, outcome = declared or "disqualified", "rejected"
    elif flags:
        verdict_class, outcome = declared or "disqualified", "flagged"
    elif declared:
        verdict_class, outcome = declared, declared
    else:
        verdict_class, outcome = "partial", "unclassified"
    return {
        **base,
        "artifact_present": True,
        "declared_verdict_class": declared,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
        "outcome": outcome,
        "flag_kinds": flags,
        "eligible_for_positive": verdict_class == "positive" and outcome == "positive",
    }


def blocked_diagnostic(
    task: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    task_row: Mapping[str, Any],
) -> JsonDict:
    """Retain exact failure values so a block stays diagnosable."""

    base = {"number": int(task["number"]), "task_id": str(task["task_id"])}
    if payload is None:
        return {
            **base,
            "failed_check": "expected_deliverable_readable",
            "expected_value": True,
            "observed_value": False,
            "gate_check_summary": "expected science artifact is absent",
            "terminal": True,
        }
    summary = payload.get("gate_check_summary")
    summary_map = summary if isinstance(summary, Mapping) else {}
    upstream = payload.get("failed_upstream")
    field = payload.get("failed_field")
    failed_check = (
        f"{upstream}.{field}"
        if upstream and field
        else summary_map.get("failed_check", "artifact_terminal_gate")
    )
    return {
        **base,
        "failed_check": failed_check,
        "expected_value": payload.get("failed_expected", summary_map.get("expected_value", 1)),
        "observed_value": payload.get(
            "failed_observed", summary_map.get("observed_value", task_row.get("verdict_class"))
        ),
        "gate_check_summary": summary or "blocked artifact",
        "terminal": True,
    }


def _load_json(path: Path) -> tuple[JsonDict | None, str | None]:
    """Read one object and distinguish absence from unreadable evidence."""

    if not path.is_file():
        return None, "absent"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "JSON root is not an object"
    return payload, None


def _state_row(number: int, task_rows: Mapping[int, Mapping[str, Any]], evidence: str) -> JsonDict:
    """Create one terminal unavailable row for an external task state."""

    task = task_rows.get(number, {})
    return {
        "evidence": evidence,
        "source_experiment": number,
        "state": task.get("outcome", task.get("verdict_class", "blocked")),
        "verdict_class": task.get("verdict_class", "blocked"),
        "available": False,
        "terminal": True,
    }


def build_evidence_isolation_rows(payloads: Mapping[int, Mapping[str, Any]]) -> list[JsonDict]:
    """Recompute learner isolation and the prohibited shortcut release gate."""

    sidecar = payloads.get(6997, {})
    audit = payloads.get(6999, {})
    tensor_rows = sidecar.get("tensor_hash_rows", [])
    tensor_hashes = {row.get("tensor_hash") for row in tensor_rows}
    prediction_hashes = {row.get("prediction_hash") for row in tensor_rows}
    rows: list[JsonDict] = [
        {
            "evidence": "learner_sidecar_summary",
            "candidate_count": len(sidecar.get("per_candidate_rows", [])),
            "source_model_row_count": len(sidecar.get("source_model_rows", [])),
            "tensor_view_count": len(tensor_rows),
            "tensor_hash_invariant": len(tensor_hashes) == 1 and bool(tensor_rows),
            "prediction_hash_invariant": len(prediction_hashes) == 1 and bool(tensor_rows),
            "prohibited_probe_count": len(sidecar.get("prohibited_feature_rows", [])),
            "prohibited_access_rejected": all(
                row.get("passed") is False for row in sidecar.get("prohibited_feature_rows", [])
            ),
            "reported_ready_score": sidecar.get("blinded_learner_view_ready_score"),
            "terminal": True,
        }
    ]
    shortcut_rows = audit.get("shortcut_interval_rows", [])
    failed = []
    for source in shortcut_rows:
        upper = source.get("ci95_upper")
        threshold = source.get("threshold", 0.8)
        prohibited = source.get("prohibited") is True
        passed = bool(
            not prohibited
            or (
                isinstance(upper, (int, float))
                and isinstance(threshold, (int, float))
                and upper < threshold
            )
        )
        name = str(source.get("probe_name") or "unnamed")
        if not passed:
            failed.append(name)
        rows.append(
            {
                "evidence": f"shortcut_probe:{name}",
                "prohibited": prohibited,
                "auroc": source.get("shortcut_auroc", source.get("auroc")),
                "ci95_upper": upper,
                "threshold": threshold,
                "recomputed_passed": passed,
                "reported_passed": source.get("gate_passed"),
                "terminal": True,
            }
        )
    invariance = audit.get("learner_invariance_rows", [])
    direct_leakage = len(audit.get("prohibited_field_rows", []))
    invariance_passed = bool(invariance) and all(row.get("passed") is True for row in invariance)
    ready = int(direct_leakage == 0 and invariance_passed and not failed)
    rows.append(
        {
            "evidence": "cold_audit_summary",
            "candidate_count": len(audit.get("per_candidate_rows", [])),
            "direct_leakage_count": direct_leakage,
            "invariance_condition_count": len(invariance),
            "invariance_passed": invariance_passed,
            "failed_prohibited_probes": failed,
            "recomputed_ready_score": ready,
            "reported_ready_score": audit.get("blinded_feature_bank_ready_score"),
            "reported_matches": ready == audit.get("blinded_feature_bank_ready_score"),
            "terminal": True,
        }
    )
    return rows


def build_commitment_control_rows(payloads: Mapping[int, Mapping[str, Any]]) -> list[JsonDict]:
    """Recompute control coverage and family-level commitment intervals."""

    source = payloads.get(6998, {})
    units = source.get("per_candidate_condition_model_rows", [])
    pairs = {row.get("pair_id") for row in units if row.get("pair_id")}
    candidates = {row.get("candidate_id") for row in units if row.get("candidate_id")}
    models = {row.get("model_id") for row in units if row.get("model_id")}
    conditions = {row.get("condition") for row in units if row.get("condition")}
    complete = int(
        bool(units)
        and all(
            row.get("terminal") is True
            and row.get("choice_extraction_passed") is True
            and row.get("tokenizer_parity_passed") is True
            for row in units
        )
    )
    rows: list[JsonDict] = [
        {
            "evidence": "commitment_summary",
            "row_count": len(units),
            "pair_count": len(pairs),
            "candidate_count": len(candidates),
            "model_count": len(models),
            "condition_count": len(conditions),
            "recomputed_complete_score": complete,
            "reported_complete_score": source.get("commitment_control_complete_score"),
            "terminal": True,
        }
    ]
    rows.extend(
        {
            "evidence": "family_commitment_interval",
            **deepcopy(row),
            "terminal": True,
        }
        for row in source.get("family_stratum_rows", [])
    )
    return rows


def _finite(values: Sequence[Any]) -> list[float]:
    """Keep finite real values and reject booleans from metric arithmetic."""

    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def build_pwa_rows(
    payloads: Mapping[int, Mapping[str, Any]],
    task_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Recompute PWA energy separation and MILP certificate outcomes."""

    source = payloads.get(7000)
    if not source or task_rows.get(7000, {}).get("verdict_class") == "blocked":
        return (
            [_state_row(7000, task_rows, "pwa_energy_unavailable")],
            [_state_row(7000, task_rows, "milp_certificates_unavailable")],
        )
    units = source.get("pwa_energy_rows", [])
    valid = _finite([row.get("energy") for row in units if row.get("exact_valid") is True])
    invalid = _finite([row.get("energy") for row in units if row.get("exact_valid") is False])
    energy_rows = [
        {
            "evidence": "pwa_energy_summary",
            "row_count": len(units),
            "valid_mean_energy": statistics.fmean(valid) if valid else None,
            "invalid_mean_energy": statistics.fmean(invalid) if invalid else None,
            "separation": statistics.fmean(invalid) - statistics.fmean(valid)
            if valid and invalid
            else None,
            "terminal": True,
        }
    ]
    certificates = source.get("milp_certificate_rows", source.get("milp_query_rows", []))
    milp_rows = [
        {**deepcopy(row), "recomputed_passed": row.get("passed") is True, "terminal": True}
        for row in certificates
    ]
    if not milp_rows:
        milp_rows = [_state_row(7000, task_rows, "milp_certificates_unavailable")]
    return energy_rows, milp_rows


def build_selection_rows(
    payloads: Mapping[int, Mapping[str, Any]],
    task_rows: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    """Recompute prospective selection rates without treating the oracle as an arm."""

    source = payloads.get(7002)
    certificate = payloads.get(7001, {})
    qualified = science_positive_score(source or {}, certificate)
    qualification = {
        "evidence": "science_qualification",
        "selection_positive_score": (source or {}).get(
            "oracle_distinct_selection_positive_score", 0
        ),
        "certificate_confirmed_score": certificate.get("pwa_certificate_confirmed_score", 0),
        "qualified_science_positive_score": qualified,
        "terminal": True,
    }
    if not source:
        return [_state_row(7002, task_rows, "selection_unavailable"), qualification]
    units = source.get("selection_evidence_rows", source.get("selection_rows", []))
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in units:
        arm = row.get("arm")
        correct = row.get("selected_correct")
        if arm and isinstance(correct, bool) and "oracle" not in str(arm).lower():
            grouped[str(arm)].append(correct)
    arm_rows = [
        {
            "evidence": "selection_arm",
            "arm": arm,
            "event_count": len(values),
            "selection_rate": sum(values) / len(values),
            "terminal": True,
        }
        for arm, values in sorted(grouped.items())
    ]
    pwa_rate = next(
        (row["selection_rate"] for row in arm_rows if row["arm"] in {"pwa_kan", "pwa-kan"}),
        None,
    )
    controls = [
        row["selection_rate"] for row in arm_rows if row["arm"] not in {"pwa_kan", "pwa-kan"}
    ]
    best_control = max(controls) if controls else None
    summary = {
        "evidence": "selection_summary",
        "pwa_kan_rate": pwa_rate,
        "best_control_rate": best_control,
        "pwa_delta": pwa_rate - best_control
        if isinstance(pwa_rate, (int, float)) and isinstance(best_control, (int, float))
        else None,
        "terminal": True,
    }
    return [*arm_rows, summary, qualification]


def classify_learning_positive(payload: Mapping[str, Any]) -> JsonDict:
    """Mark verifier-released learning as circular without outside authority."""

    positive = payload.get("per_knot_learning_positive_score") == 1
    independent = payload.get("independent_outcome_authority") is True
    verdict_class = (
        "positive" if positive and independent else ("circular_positive" if positive else "null")
    )
    return {
        "verdict_class": verdict_class,
        "independent_outcome_authority": independent,
        "science_positive": False,
    }


def build_learning_rows(
    payloads: Mapping[int, Mapping[str, Any]],
    task_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Recompute future gain, forgetting, and best-at-k support."""

    source = payloads.get(7003)
    if not source:
        return (
            [_state_row(7003, task_rows, "continuous_learning_unavailable")],
            [_state_row(7004, task_rows, "support_and_forgetting_unavailable")],
        )
    units = source.get("continuous_learning_rows", source.get("rows", []))
    target = [row for row in units if row.get("arm") == "per_knot_anchor"]
    future_gain = sum(
        int(row.get("future_correct_after", 0)) - int(row.get("future_correct_before", 0))
        for row in target
    )
    lost = sum(row.get("prior_success_lost") is True for row in target)
    support_delta = sum(
        int(row.get("best_at_k_after", 0)) - int(row.get("best_at_k_before", 0)) for row in target
    )
    classification = classify_learning_positive(source)
    learning = [
        {
            "evidence": "per_knot_learning_summary",
            "row_count": len(target),
            "future_gain": future_gain,
            **classification,
            "terminal": True,
        }
    ]
    support = [
        {
            "evidence": "per_knot_support_summary",
            "prior_successes_lost": lost,
            "best_at_k_support_delta": support_delta,
            "audit_confirmed_score": payloads.get(7004, {}).get(
                "self_learning_audit_confirmed_score"
            ),
            "terminal": True,
        }
    ]
    return learning, support


def build_arc_rows(
    payloads: Mapping[int, Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Recompute ARC envelope hashes and held-out transition quality."""

    source = payloads.get(7005, {})
    hash_fields = (
        "envelope_hash_replay_rows",
        "prompt_hash_replay_rows",
        "transition_hash_replay_rows",
        "engine_hash_replay_rows",
        "environment_hash_replay_rows",
        "policy_hash_replay_rows",
        "factory_hash_replay_rows",
        "manifest_hash_replay_rows",
        "scorer_hash_replay_rows",
    )
    hash_rows = [row for field in hash_fields for row in source.get(field, [])]
    envelope_rows = [
        {
            "evidence": "hash_replay",
            "envelope_count": len(source.get("envelope_hash_replay_rows", [])),
            "transition_count": len(source.get("transition_hash_replay_rows", [])),
            "hash_check_count": len(hash_rows),
            "all_hashes_passed": bool(hash_rows)
            and all(row.get("passed") is True for row in hash_rows),
            "selected_envelope_hash": source.get("selected_envelope_hash"),
            "development_fixture_used_for_quality": source.get(
                "development_fixture_used_for_quality"
            ),
            "terminal": True,
        }
    ]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in source.get("per_transition_rows", []):
        if row.get("target_available") is True:
            grouped[(str(row.get("predictor")), str(row.get("stratum")))].append(row)
    quality_rows: list[JsonDict] = []
    for (predictor, stratum), units in sorted(grouped.items()):
        errors = _finite([row.get("calibrated_frame_error") for row in units])
        exact = [row.get("exact_next_frame_correct") is True for row in units]
        quality_rows.append(
            {
                "evidence": "transition_quality",
                "predictor": predictor,
                "stratum": stratum,
                "transition_count": len(units),
                "exact_next_frame_accuracy": sum(exact) / len(exact),
                "calibrated_frame_error": statistics.fmean(errors),
                "terminal": True,
            }
        )
    heldout = {row["predictor"]: row for row in quality_rows if row["stratum"] == "heldout"}
    engine = heldout.get("engine", {})
    controls = [
        heldout.get("inert_no_change", {}),
        heldout.get("observed_action_delta_hypothesis", {}),
    ]
    positive = bool(
        engine
        and all(control for control in controls)
        and all(
            engine.get("calibrated_frame_error", float("inf"))
            < control.get("calibrated_frame_error", float("inf"))
            and engine.get("exact_next_frame_accuracy", 0)
            > control.get("exact_next_frame_accuracy", 0)
            for control in controls
        )
    )
    quality_rows.append(
        {
            "evidence": "heldout_engine_quality_summary",
            "recomputed_positive_score": int(positive),
            "reported_positive_score": source.get("arc_engine_quality_positive_score"),
            "solve_claimed": False,
            "level_claimed": False,
            "registry_updated": False,
            "terminal": True,
        }
    )
    return envelope_rows, quality_rows


def build_z1t_rows(
    payloads: Mapping[int, Mapping[str, Any]],
    task_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Recompute static graph and partition counts without a device claim."""

    source = payloads.get(7006)
    if not source:
        return (
            [_state_row(7006, task_rows, "z1t_typed_graph_unavailable")],
            [_state_row(7007, task_rows, "z1t_placement_unavailable")],
        )
    nodes = source.get("z1t_typed_graph_rows", source.get("typed_graph_rows", []))
    placements = source.get("z1t_placement_rows", source.get("placement_rows", []))
    graph = [
        {
            "evidence": "typed_graph_summary",
            "node_count": len(nodes),
            "z1_native_count": sum(row.get("placement") == "z1_native" for row in nodes),
            "unsupported_count": sum(row.get("placement") == "unsupported" for row in nodes),
            "terminal": True,
        }
    ]
    placement = [
        {
            "evidence": "placement_summary",
            "placement_row_count": len(placements),
            "crossing_count": sum(row.get("crossing") is True for row in placements),
            "cold_audit_confirmed_score": payloads.get(7007, {}).get(
                "z1t_placement_confirmed_score"
            ),
            "hardware_executed": False,
            "terminal": True,
        }
    ]
    return graph, placement


def science_positive_score(
    selection_payload: Mapping[str, Any], certificate_payload: Mapping[str, Any]
) -> int:
    """Admit only oracle-distinct selection with a confirmed certificate."""

    return int(
        selection_payload.get("oracle_distinct_selection_positive_score") == 1
        and certificate_payload.get("pwa_certificate_confirmed_score") == 1
    )


def publication_state(payload: Mapping[str, Any]) -> JsonDict:
    """Copy the fixed publication gate rather than deriving it from tasks."""

    gates = payload.get("gates", {})
    rows = [
        {
            "gate": gate,
            "passed": bool(gates.get(gate, {}).get("pass")),
            "detail": gates.get(gate, {}).get("detail"),
            "terminal": True,
        }
        for gate in ("G1", "G2", "G3", "G4")
    ]
    return {
        "publication_gate_rows": rows,
        "g1": rows[0]["passed"],
        "g2": rows[1]["passed"],
        "g3": rows[2]["passed"],
        "g4": rows[3]["passed"],
        "paper_ready": bool(payload.get("paper_ready")),
        "unmet_gates": list(payload.get("unmet_gates", [])),
    }


def build_retirement_rows(
    yaml_rows: Sequence[Mapping[str, Any]],
    payloads: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    """Retire only a verdict that exactly repeats a declared prior verdict."""

    rows = []
    for task in yaml_rows:
        current = str(payloads.get(int(task.get("number", -1)), {}).get("honest_verdict") or "")
        for prior in task.get("prior_failures", []):
            prior_verdict = str(prior.get("verdict") or "")
            exact = bool(current and current == prior_verdict)
            retired = bool(prior.get("retire_if_same_verdict") is True and exact)
            rows.append(
                {
                    "number": task.get("number"),
                    "task_id": task.get("task_id"),
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior_verdict,
                    "current_verdict": current or "artifact_missing",
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "exact_same_verdict": exact,
                    "retired": retired,
                    "recommendation": "no_rerun_without_new_cause_and_method"
                    if retired
                    else "new_cause_and_method_required_for_any_rerun",
                    "terminal": True,
                }
            )
    return rows


def _sha256(path: Path) -> str:
    """Hash source bytes so later audits can detect evidence replacement."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _field_principles() -> dict[str, str]:
    """Give every artifact field an explicit evidence principle."""

    return {
        field: f"The {field} field must remain traceable to deterministic evidence."
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _placeholder_command_rows() -> list[JsonDict]:
    """Keep library builds terminal while the CLI owns external checks."""

    return [
        {
            "name": name,
            "command": "deferred_to_cli",
            "exit_code": None,
            "finding": "not_run_in_library_build",
            "terminal": True,
        }
        for name in COMMAND_NAMES
    ]


def _publication_payload(root: Path) -> JsonDict:
    """Run the stable publication evaluator and fail closed on bad output."""

    result = subprocess.run(
        [str(root / ".venv/bin/python"), "scripts/publication_gate.py", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict) and isinstance(payload.get("gates"), dict):
        return payload
    return {
        "paper_ready": False,
        "gates": {
            gate: {"pass": False, "detail": "publication gate unreadable"}
            for gate in ("G1", "G2", "G3", "G4")
        },
        "unmet_gates": ["G1", "G2", "G3", "G4"],
    }


def _preconditions(root: Path) -> tuple[list[JsonDict], bool]:
    """Check only inputs required to execute the evidence reconciliation."""

    required = (
        ROADMAP_PATH,
        DESIGN_PATH,
        V612_CAPSTONE_PATH,
        EXCLUSION_PATH,
        ARC_REGISTRY_PATH,
        Path("scripts/publication_gate.py"),
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
        Path("scripts/recurring_blocker_ledger.py"),
        Path("scripts/root_clutter_sweep.py"),
        Path("scripts/check_spec_coverage.py"),
    )
    rows = []
    for relative in required:
        path = root / relative
        try:
            readable = path.is_file() and bool(path.read_bytes())
        except OSError:
            readable = False
        rows.append(
            {
                "check": f"readable:{relative.as_posix()}",
                "expected_value": True,
                "observed_value": readable,
                "passed": readable,
                "terminal": True,
            }
        )
    results_readable = (root / "results").is_dir()
    rows.append(
        {
            "check": "readable:results_directory",
            "expected_value": True,
            "observed_value": results_readable,
            "passed": results_readable,
            "terminal": True,
        }
    )
    return rows, all(row["passed"] for row in rows)


def _empty_artifact(run_date: str, publication: Mapping[str, Any]) -> JsonDict:
    """Return a schema-complete base used by both blocked and complete runs."""

    artifact: JsonDict = {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field.endswith("_rows") or field == "rows"
    }
    artifact.update(
        {
            "schema": "carnot.v613-capstone.v1",
            "experiment_id": 7008,
            "run_date": run_date,
            "status": "blocked",
            "field_principles": _field_principles(),
            "preconditions_checked": [],
            "inference_substrate": INFERENCE_SUBSTRATE,
            "duration_s": 0.0,
            "source_artifact_hashes": {},
            "expected_task_rows": [{**deepcopy(task), "terminal": True} for task in EXPECTED_TASKS],
            "missing_task_rows": [],
            "g1": publication["g1"],
            "g2": publication["g2"],
            "g3": publication["g3"],
            "g4": publication["g4"],
            "paper_ready": publication["paper_ready"],
            "unmet_gates": publication["unmet_gates"],
            "publication_gate_rows": publication["publication_gate_rows"],
            "command_receipt_rows": _placeholder_command_rows(),
            "v613_capstone_complete_score": 0,
            "v613_task_contract_conforms_score": 0,
            "v613_science_positive_score": 0,
            "solve_claimed": False,
            "level_claimed": False,
            "registry_updated": False,
            "hardware_executed": False,
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "gate_check_summary": {},
            "verifier_is_oracle": True,
            "verdict_class": "blocked",
            "honest_verdict": BLOCKED_VERDICT,
        }
    )
    return artifact


def _terminal_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require each receipt in a completed collection to be terminal."""

    return bool(rows) and all(row.get("terminal") is True for row in rows)


def _complete_score(artifact: Mapping[str, Any]) -> int:
    """Recompute completion from task, contract, branch, gate, and command rows."""

    names = (
        "rows",
        "task_contract_rows",
        "gate_contract_rows",
        "gate_producer_rows",
        "prior_failure_rows",
        "per_branch_results",
        "publication_gate_rows",
        "retirement_rows",
        "command_receipt_rows",
    )
    return int(
        artifact.get("status") != "blocked"
        and all(_terminal_rows(artifact.get(name, [])) for name in names)
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding wall time and command prose."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s", "command_receipt_rows"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_artifact(
    root: Path,
    run_date: str,
    *,
    run_commands: bool,
    publication_payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the capstone without modifying any source artifact."""

    del run_commands
    started = time.monotonic()
    publication = publication_state(publication_payload or _publication_payload(root))
    preconditions, ready = _preconditions(root)
    artifact = _empty_artifact(run_date, publication)
    artifact["preconditions_checked"] = preconditions
    available_preconditions = [
        ROADMAP_PATH,
        DESIGN_PATH,
        V612_CAPSTONE_PATH,
        EXCLUSION_PATH,
        ARC_REGISTRY_PATH,
    ]
    artifact["source_artifact_hashes"] = {
        path.as_posix(): _sha256(root / path)
        for path in available_preconditions
        if (root / path).is_file()
    }
    if not ready:
        artifact["gate_check_summary"] = {
            "failed_check": "contract_preconditions",
            "expected_value": "readable activated V613 contracts",
            "observed_value": "missing or unreadable",
            "passed": False,
        }
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    try:
        roadmap = load_yaml(root / ROADMAP_PATH)
        document_rows = parse_document_contract((root / DESIGN_PATH).read_text(encoding="utf-8"))
        yaml_rows = parse_yaml_contract(roadmap)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError):
        artifact["gate_check_summary"] = {
            "failed_check": "contract_preconditions",
            "expected_value": "readable activated V613 contracts",
            "observed_value": "missing or unreadable",
            "passed": False,
        }
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    task_contract = build_task_contract_rows(document_rows, yaml_rows)
    gate_contract = build_gate_contract_rows(document_rows, yaml_rows)
    producer_rows = build_gate_producer_rows(yaml_rows)
    prior_rows = build_prior_failure_rows(yaml_rows)
    conforms = contract_conforms(
        document_rows, yaml_rows, task_contract, gate_contract, producer_rows, prior_rows
    )

    payloads: dict[int, JsonDict] = {}
    source_errors: list[JsonDict] = []
    task_rows: list[JsonDict] = []
    for task in EXPECTED_TASKS[:-1]:
        number = int(task["number"])
        path = root / str(task["deliverable"])
        payload, error = _load_json(path)
        if payload is not None:
            payloads[number] = payload
            artifact["source_artifact_hashes"][str(task["deliverable"])] = _sha256(path)
        if error not in {None, "absent"}:
            source_errors.append(
                {
                    "number": number,
                    "source": str(task["deliverable"]),
                    "recomputed": "unreadable",
                    "reported": error,
                    "terminal": True,
                }
            )
        task_rows.append(classify_task(task, payload))
    task_by_number = {int(row["number"]): row for row in task_rows}

    isolation_rows = build_evidence_isolation_rows(payloads)
    commitment_rows = build_commitment_control_rows(payloads)
    pwa_rows, milp_rows = build_pwa_rows(payloads, task_by_number)
    selection_rows = build_selection_rows(payloads, task_by_number)
    learning_rows, support_rows = build_learning_rows(payloads, task_by_number)
    arc_envelopes, arc_quality = build_arc_rows(payloads)
    z1t_graph, z1t_placement = build_z1t_rows(payloads, task_by_number)
    science = science_positive_score(payloads.get(7002, {}), payloads.get(7001, {}))
    verdict_class = "positive" if science else "null"
    honest_verdict = POSITIVE_VERDICT if science else NULL_VERDICT
    self_row = classify_task(
        EXPECTED_TASKS[-1],
        {
            "status": "complete",
            "verdict_class": verdict_class,
            "honest_verdict": honest_verdict,
        },
    )
    task_rows.append(self_row)
    task_by_number[7008] = self_row

    blocked_rows = [
        blocked_diagnostic(
            task, payloads.get(int(task["number"])), task_by_number[int(task["number"])]
        )
        for task in EXPECTED_TASKS[:-1]
        if task_by_number[int(task["number"])]["verdict_class"] == "blocked"
    ]
    flagged_rows = [
        {
            "number": row["number"],
            "task_id": row["task_id"],
            "flag_kinds": row["flag_kinds"],
            "excluded_from_positive": True,
            "terminal": True,
        }
        for row in task_rows
        if row.get("outcome") == "flagged"
    ]

    failed_probes = next(
        (
            row["failed_prohibited_probes"]
            for row in isolation_rows
            if row.get("evidence") == "cold_audit_summary"
        ),
        [],
    )
    per_branch = [
        {
            "branch": "evidence_isolation",
            "verdict_class": task_by_number[6999]["verdict_class"],
            "state": task_by_number[6999]["outcome"],
            "pooled_with_other_branches": False,
            "terminal": True,
        },
        {
            "branch": "oracle_distinct_learned_selection",
            "verdict_class": task_by_number[7002]["verdict_class"],
            "state": task_by_number[7002]["outcome"],
            "pooled_with_other_branches": False,
            "terminal": True,
        },
        {
            "branch": "verifier_grounded_continuous_self_learning",
            "verdict_class": task_by_number[7003]["verdict_class"],
            "state": task_by_number[7003]["outcome"],
            "pooled_with_other_branches": False,
            "terminal": True,
        },
        {
            "branch": "arc_live_envelope_engine_quality",
            "verdict_class": task_by_number[7005]["verdict_class"],
            "state": task_by_number[7005]["outcome"],
            "pooled_with_other_branches": False,
            "terminal": True,
        },
        {
            "branch": "static_z1t_placement",
            "verdict_class": task_by_number[7007]["verdict_class"],
            "state": task_by_number[7007]["outcome"],
            "pooled_with_other_branches": False,
            "terminal": True,
        },
    ]
    retirement_rows = build_retirement_rows(yaml_rows, payloads)
    circularity = [
        {
            "source_experiment": 6997,
            "declared_verdict_class": task_by_number[6997]["verdict_class"],
            "science_positive": False,
            "reason": "sidecar completion is infrastructure evidence",
            "terminal": True,
        },
        {
            "source_experiment": 7003,
            "declared_verdict_class": task_by_number[7003]["verdict_class"],
            "science_positive": False,
            "reason": "per-knot positives need independent outcome authority",
            "terminal": True,
        },
    ]
    boundaries = [
        {
            "order": 1,
            "boundary": "exp6999_prohibited_feature_shortcut_gate",
            "state": task_by_number[6999]["outcome"],
            "failed_probes": failed_probes,
            "earliest": True,
            "terminal": True,
        },
        {
            "order": 2,
            "boundary": "exp7000_pwa_gate_cascade",
            "state": task_by_number[7000]["outcome"],
            "earliest": False,
            "terminal": True,
        },
        {
            "order": 3,
            "boundary": "exp7005_arc_engine_quality",
            "state": task_by_number[7005]["outcome"],
            "earliest": False,
            "terminal": True,
        },
    ]
    forbidden = [
        "spilled_energy",
        "schema_only_reprompt",
        "finite_id_answer_transport",
        "exp5895_exact_slot",
        "public_arc_resolve",
        "unchanged_fpga_probe",
        "hardware_execution_without_device_receipt",
    ]
    handoff = [
        {
            "earliest_causal_boundary": "exp6999_prohibited_feature_shortcut_gate",
            "recommendation": (
                "V614 must break the authority-sidecar correlation for "
                + ", ".join(failed_probes)
                + " with a new cause and method before fitting another ranker."
            ),
            "forbidden_revivals": forbidden,
            "preserved_branch_states": [row["state"] for row in per_branch],
            "terminal": True,
        }
    ]

    source_disagreements = source_errors
    for row in isolation_rows:
        if row.get("reported_matches") is False:
            source_disagreements.append(
                {
                    "source": "experiment_6999",
                    "field": "blinded_feature_bank_ready_score",
                    "recomputed": row.get("recomputed_ready_score"),
                    "reported": row.get("reported_ready_score"),
                    "terminal": True,
                }
            )

    artifact.update(
        {
            "status": "complete",
            "rows": task_rows,
            "observed_task_rows": [row for row in task_rows if row["artifact_present"]],
            "missing_task_rows": [row for row in task_rows if row["outcome"] == "missing"],
            "task_contract_rows": task_contract,
            "gate_contract_rows": gate_contract,
            "gate_producer_rows": producer_rows,
            "prior_failure_rows": prior_rows,
            "verdict_class_rows": [
                {
                    "number": row["number"],
                    "task_id": row["task_id"],
                    "verdict_class": row["verdict_class"],
                    "outcome": row["outcome"],
                    "terminal": True,
                }
                for row in task_rows
            ],
            "blocked_diagnostic_rows": blocked_rows,
            "flagged_artifact_rows": flagged_rows,
            "per_branch_results": per_branch,
            "evidence_isolation_rows": isolation_rows,
            "commitment_control_rows": commitment_rows,
            "pwa_energy_rows": pwa_rows,
            "milp_certificate_rows": milp_rows,
            "selection_evidence_rows": selection_rows,
            "continuous_learning_rows": learning_rows,
            "support_and_forgetting_rows": support_rows,
            "arc_live_envelope_rows": arc_envelopes,
            "arc_quality_rows": arc_quality,
            "z1t_typed_graph_rows": z1t_graph,
            "z1t_placement_rows": z1t_placement,
            "circularity_rows": circularity,
            "claim_boundary_rows": boundaries,
            "source_disagreement_rows": source_disagreements,
            "retirement_rows": retirement_rows,
            "v614_handoff_rows": handoff,
            "v613_task_contract_conforms_score": int(conforms),
            "v613_science_positive_score": science,
            "verdict_class": verdict_class,
            "honest_verdict": honest_verdict,
            "gate_check_summary": {
                "failed_check": None if science else "oracle_distinct_selection_positive",
                "expected_value": 1,
                "observed_value": science,
                "passed": bool(science),
                "contract_conforms": conforms,
                "terminal_task_count": len(task_rows),
            },
        }
    )
    artifact["v613_capstone_complete_score"] = _complete_score(artifact)
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject schema, score, publication, claim, prefix, and checksum drift."""

    errors = [
        f"missing required field: {field}"
        for field in sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    ]
    if set(artifact.get("field_principles", {})) != REQUIRED_ARTIFACT_FIELDS:
        errors.append("field principles do not cover every required field")
    expected_complete = _complete_score(artifact)
    if artifact.get("v613_capstone_complete_score") != expected_complete:
        errors.append("complete score does not match terminal rows")
    qualification = next(
        (
            row
            for row in artifact.get("selection_evidence_rows", [])
            if row.get("evidence") == "science_qualification"
        ),
        {},
    )
    expected_science = int(qualification.get("qualified_science_positive_score") == 1)
    if artifact.get("v613_science_positive_score") != expected_science:
        errors.append("science score does not match confirmed selection")
    gate_rows = artifact.get("publication_gate_rows", [])
    for index, name in enumerate(("g1", "g2", "g3", "g4")):
        expected = gate_rows[index].get("passed") if index < len(gate_rows) else None
        if artifact.get(name) != expected:
            errors.append(f"publication gate {name} does not match source row")
    gate_passes = [row.get("passed") is True for row in gate_rows]
    if artifact.get("paper_ready") != all(gate_passes):
        errors.append("paper_ready does not match stable gates")
    expected_unmet = [row.get("gate") for row in gate_rows if row.get("passed") is not True]
    if artifact.get("unmet_gates") != expected_unmet:
        errors.append("unmet_gates do not match stable gates")
    if any(
        artifact.get(field) is not False
        for field in ("solve_claimed", "level_claimed", "registry_updated", "hardware_executed")
    ):
        errors.append("claim boundary requires false solve, level, registry, and hardware fields")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict") or "")
    prefix_ok = {
        "positive": verdict.startswith("complete_positive_"),
        "null": verdict.startswith("complete_null_"),
        "blocked": verdict.startswith("blocked_"),
        "disqualified": verdict.startswith("complete_disqualified_"),
        "circular_positive": verdict.startswith(
            ("complete_circular_positive_", "circular_positive:")
        ),
        "partial": verdict.startswith("partial_"),
    }.get(verdict_class, False)
    if not prefix_ok:
        errors.append("honest verdict prefix does not match verdict class")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def run_command(root: Path, name: str, command: Sequence[str]) -> JsonDict:
    """Run one bounded checker and retain both output channels as evidence."""

    result = subprocess.run(list(command), cwd=root, capture_output=True, text=True, check=False)
    finding = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return {
        "name": name,
        "command": shlex.join(command),
        "exit_code": result.returncode,
        "finding": finding[:8000],
        "terminal": True,
    }


def run_required_commands(root: Path, output: Path) -> list[JsonDict]:
    """Run every read-only capstone check exactly once after provisional output."""

    python = str(root / ".venv/bin/python")
    script = str(root / "scripts/experiments/experiment_7008_v613_capstone.py")
    commands = (
        ("publication_gate", [python, "scripts/publication_gate.py", "--json"]),
        ("artifact_validation", [python, script, "--validate", str(output)]),
        ("adversarial_verification", [python, "scripts/adversarial_verify.py", str(output)]),
        ("row_consistency_lint", [python, "scripts/verdict_row_consistency_lint.py", str(output)]),
        (
            "recurring_blocker_ledger",
            [python, "scripts/recurring_blocker_ledger.py", "--window", "14"],
        ),
        ("openspec_coverage", [python, "scripts/check_spec_coverage.py"]),
        ("root_clutter_check", [python, "scripts/root_clutter_sweep.py"]),
    )
    return [run_command(root, name, command) for name, command in commands]


def _write(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write only the caller-selected output path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate one V613 capstone artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260905")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-commands", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        payload, error = _load_json(args.validate)
        errors = [error] if error else validate_artifact(payload or {})
        if errors:
            print(json.dumps({"valid": False, "errors": errors}, indent=2))
            return 1
        print(json.dumps({"valid": True, "errors": []}, indent=2))
        return 0

    root = args.root.resolve()
    output = args.output or (root / OUTPUT_PATH)
    if not output.is_absolute():
        output = root / output
    artifact = build_artifact(root, args.date, run_commands=False)
    _write(output, artifact)
    if not args.no_commands and artifact.get("status") != "blocked":
        artifact["command_receipt_rows"] = run_required_commands(root, output)
        artifact["v613_capstone_complete_score"] = _complete_score(artifact)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        _write(output, artifact)
    errors = validate_artifact(artifact)
    print(json.dumps({"artifact": str(output), "valid": not errors, "errors": errors}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - exercised through the thin script entrypoint.
    raise SystemExit(main(sys.argv[1:]))
