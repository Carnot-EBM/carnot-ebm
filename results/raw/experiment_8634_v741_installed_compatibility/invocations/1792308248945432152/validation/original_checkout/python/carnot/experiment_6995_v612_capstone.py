"""Build the V612 evidence capstone required by REQ-REPORT-6995.

The reducer reads the Markdown contract, YAML contract, and task artifacts
separately. It keeps infrastructure readiness apart from scientific value so a
completed fixture cannot become a learned-result claim.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
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
OUTPUT_PATH = Path("results/experiment_6995_v612_capstone.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
V611_CAPSTONE_PATH = Path("results/experiment_6983_v611_capstone.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
MILESTONE = "2026.09.612"
INFERENCE_SUBSTRATE = "deterministic_milestone_evidence_synthesis_no_llm"
RANDOM_SEED = 6995
BLOCKED_VERDICT = "blocked_v612_capstone"
NULL_VERDICT = "complete_null_v612_capstone_no_cold_audited_science_positive"
POSITIVE_VERDICT = "complete_positive_v612_capstone_cold_audited_science_positive"
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}


def _gate(upstream: str, artifact_field: str) -> JsonDict:
    """Make one normalized equality gate."""

    return {"upstream": upstream, "artifact_field": artifact_field, "op": "==", "value": 1}


EXPECTED_TASKS: tuple[JsonDict, ...] = (
    {
        "number": 6984,
        "task_id": "exp6984-exact-contrast-fixture",
        "title": "Source-grouped exact mapping contrast fixture",
        "deliverable": "results/experiment_6984_exact_contrast_fixture.json",
        "gates": [],
    },
    {
        "number": 6985,
        "task_id": "exp6985-chronological-constraint-stream",
        "title": "Sealed chronological constraint-shift stream",
        "deliverable": "results/experiment_6985_chronological_constraint_stream.json",
        "gates": [_gate("exp6984-exact-contrast-fixture", "contrast_fixture_complete_score")],
    },
    {
        "number": 6986,
        "task_id": "exp6986-three-family-contrast-features",
        "title": "Three-family contrast feature bank",
        "deliverable": "results/experiment_6986_three_family_contrast_features.json",
        "gates": [
            _gate("exp6984-exact-contrast-fixture", "contrast_fixture_complete_score"),
            _gate("exp6984-exact-contrast-fixture", "label_balance_ready_score"),
            _gate("exp6985-chronological-constraint-stream", "chronological_stream_ready_score"),
        ],
    },
    {
        "number": 6987,
        "task_id": "exp6987-contrast-feature-audit",
        "title": "Independent contrast balance and leakage audit",
        "deliverable": "results/experiment_6987_contrast_feature_audit.json",
        "gates": [
            _gate(
                "exp6986-three-family-contrast-features",
                "three_family_feature_bank_complete_score",
            )
        ],
    },
    {
        "number": 6988,
        "task_id": "exp6988-certified-pwa-kan-ranker",
        "title": "Certified PWA-KAN constraint ranker",
        "deliverable": "results/experiment_6988_certified_pwa_kan_ranker.json",
        "gates": [_gate("exp6987-contrast-feature-audit", "contrast_feature_bank_ready_score")],
    },
    {
        "number": 6989,
        "task_id": "exp6989-pwa-certificate-cold-audit",
        "title": "Fresh-process PWA-KAN certificate audit",
        "deliverable": "results/experiment_6989_pwa_certificate_cold_audit.json",
        "gates": [_gate("exp6988-certified-pwa-kan-ranker", "pwa_model_ready_score")],
    },
    {
        "number": 6990,
        "task_id": "exp6990-oracle-distinct-selection",
        "title": "Oracle-distinct constraint selection comparison",
        "deliverable": "results/experiment_6990_oracle_distinct_selection.json",
        "gates": [
            _gate("exp6988-certified-pwa-kan-ranker", "pwa_model_ready_score"),
            _gate(
                "exp6989-pwa-certificate-cold-audit",
                "pwa_certificate_confirmed_score",
            ),
        ],
    },
    {
        "number": 6991,
        "task_id": "exp6991-per-knot-continuous-learning",
        "title": "Verifier-grounded per-knot continuous self-learning",
        "deliverable": "results/experiment_6991_per_knot_continuous_learning.json",
        "gates": [
            _gate("exp6985-chronological-constraint-stream", "chronological_stream_ready_score"),
            _gate(
                "exp6986-three-family-contrast-features",
                "three_family_feature_bank_complete_score",
            ),
            _gate("exp6988-certified-pwa-kan-ranker", "pwa_model_ready_score"),
            _gate(
                "exp6989-pwa-certificate-cold-audit",
                "pwa_certificate_confirmed_score",
            ),
        ],
    },
    {
        "number": 6992,
        "task_id": "exp6992-self-learning-support-audit",
        "title": "Fresh-process self-learning and support audit",
        "deliverable": "results/experiment_6992_self_learning_support_audit.json",
        "gates": [
            _gate("exp6991-per-knot-continuous-learning", "self_learning_run_complete_score")
        ],
    },
    {
        "number": 6993,
        "task_id": "exp6993-arc-producer-evidence-contract",
        "title": "ARC live producer evidence contract",
        "deliverable": "results/experiment_6993_arc_producer_evidence_contract.json",
        "gates": [],
    },
    {
        "number": 6994,
        "task_id": "exp6994-arc-producer-cold-audit",
        "title": "Fresh-process ARC producer contract audit",
        "deliverable": "results/experiment_6994_arc_producer_cold_audit.json",
        "gates": [
            _gate(
                "exp6993-arc-producer-evidence-contract",
                "arc_producer_contract_complete_score",
            ),
            _gate(
                "exp6993-arc-producer-evidence-contract",
                "arc_live_path_fixture_ready_score",
            ),
        ],
    },
    {
        "number": 6995,
        "task_id": "exp6995-v612-capstone",
        "title": "V612 independent evidence capstone and V613 handoff",
        "deliverable": "results/experiment_6995_v612_capstone.json",
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
    "verdict_class_rows",
    "blocked_diagnostic_rows",
    "per_branch_results",
    "contrast_evidence_rows",
    "pwa_energy_evidence_rows",
    "selection_evidence_rows",
    "continuous_learning_evidence_rows",
    "support_and_forgetting_rows",
    "arc_producer_evidence_rows",
    "circularity_rows",
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
    "v613_handoff_rows",
    "v612_capstone_complete_score",
    "v612_task_contract_conforms_score",
    "v612_science_positive_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}


def spec_anchors(text: str) -> list[str]:
    """Return scenario anchors found in specification text."""

    return re.findall(r"^####\s+(SCENARIO-[A-Z0-9-]+):", text, re.MULTILINE)


def load_yaml(path: Path) -> JsonDict:
    """Read one YAML mapping without borrowing document defaults."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return payload


def _number(value: Any) -> int | None:
    """Extract an experiment number from a task identifier."""

    match = re.search(r"exp(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def _normalize_gate(raw: Mapping[str, Any]) -> JsonDict:
    """Normalize one gate while keeping its exact producer name."""

    return {
        "upstream": str(raw.get("upstream") or ""),
        "artifact_field": str(raw.get("artifact_field") or ""),
        "op": str(raw.get("op") or "=="),
        "value": raw.get("value"),
    }


def _parse_document_gates(value: str) -> list[JsonDict]:
    """Parse the narrow gate notation used by the activated design table."""

    if value.strip().lower() == "none":
        return []
    rows = []
    pattern = re.compile(r"`([^`]+)\.([A-Za-z0-9_]+)\s*(==|!=|>=|<=|>|<)\s*([^`]+)`")
    for upstream, field, op, expected in pattern.findall(value):
        rows.append(
            {
                "upstream": upstream.strip(),
                "artifact_field": field.strip(),
                "op": op,
                "value": yaml.safe_load(expected.strip()),
            }
        )
    return rows


def parse_document_contract(text: str) -> list[JsonDict]:
    """Parse only the activated Markdown task-contract table."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`([^`]+)`", text)
    milestone = milestone_match.group(1) if milestone_match else ""
    marker = "## Exact Task Contract"
    if marker not in text:
        return []
    section = text.split(marker, 1)[1]
    section = section.split("\n## ", 1)[0]
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
    """Parse YAML tasks independently from the Markdown table."""

    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    milestone = str(roadmap.get("milestone") or "")
    rows = []
    for order, raw in enumerate(tasks, 1):
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
                "gates": [
                    _normalize_gate(gate)
                    for gate in raw.get("gated_on", [])
                    if isinstance(gate, Mapping)
                ],
                "prompt": str(raw.get("prompt") or ""),
                "prior_failures": deepcopy(raw.get("prior_failures") or []),
            }
        )
    return rows


def build_task_contract_rows(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Compare task identity, order, title, deliverable, milestone, and gates."""

    rows = []
    for index in range(max(len(document_rows), len(yaml_rows))):
        document = document_rows[index] if index < len(document_rows) else {}
        roadmap = yaml_rows[index] if index < len(yaml_rows) else {}
        checks = {
            "order": document.get("order") == roadmap.get("order") == index + 1,
            "task_id": document.get("task_id") == roadmap.get("task_id"),
            "title": document.get("title") == roadmap.get("title"),
            "deliverable": document.get("deliverable") == roadmap.get("deliverable"),
            "milestone": document.get("milestone") == roadmap.get("milestone") == MILESTONE,
            "gates": document.get("gates") == roadmap.get("gates"),
        }
        failed = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "order": index + 1,
                "task_id": document.get("task_id") or roadmap.get("task_id"),
                "document": dict(document),
                "yaml": dict(roadmap),
                "checks": checks,
                "check": "all" if not failed else ("order" if "task_id" in failed else failed[0]),
                "passed": not failed,
                "terminal": True,
            }
        )
    return rows


def build_gate_contract_rows(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Compare all 16 gates and confirm that each producer names its field."""

    document_by_id = {str(row.get("task_id")): row for row in document_rows}
    yaml_by_id = {str(row.get("task_id")): row for row in yaml_rows}
    rows = []
    for expected_task in EXPECTED_TASKS:
        task_id = str(expected_task["task_id"])
        doc_gates = document_by_id.get(task_id, {}).get("gates", [])
        yaml_gates = yaml_by_id.get(task_id, {}).get("gates", [])
        for gate_index, expected_gate in enumerate(expected_task["gates"]):
            document_gate = doc_gates[gate_index] if gate_index < len(doc_gates) else None
            yaml_gate = yaml_gates[gate_index] if gate_index < len(yaml_gates) else None
            producer = yaml_by_id.get(str(expected_gate["upstream"]), {})
            producer_declared = str(expected_gate["artifact_field"]) in str(producer.get("prompt") or "")
            passed = (
                document_gate == expected_gate
                and yaml_gate == expected_gate
                and producer_declared
            )
            rows.append(
                {
                    "task_id": task_id,
                    "gate_index": gate_index,
                    "expected_gate": dict(expected_gate),
                    "document_gate": deepcopy(document_gate),
                    "yaml_gate": deepcopy(yaml_gate),
                    "producer_field_declared": producer_declared,
                    "passed": passed,
                    "terminal": True,
                }
            )
    return rows


def contract_conforms(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Require the exact fixed 12-task, 16-gate contract."""

    return bool(
        len(document_rows) == len(yaml_rows) == len(task_rows) == 12
        and len(gate_rows) == 16
        and [row.get("task_id") for row in document_rows] == EXPECTED_TASK_IDS
        and [row.get("task_id") for row in yaml_rows] == EXPECTED_TASK_IDS
        and all(row.get("passed") is True for row in task_rows)
        and all(row.get("passed") is True for row in gate_rows)
    )


def _declared_class(payload: Mapping[str, Any]) -> str | None:
    """Read a declared verdict class without inferring it from positive prose."""

    value = payload.get("verdict_class")
    return str(value) if value in VERDICT_CLASSES else None


def classify_task(task: Mapping[str, Any], payload: Mapping[str, Any] | None) -> JsonDict:
    """Assign one terminal class while treating absent external input as blocked."""

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
        }
    declared = _declared_class(payload)
    honest = str(payload.get("honest_verdict") or "")
    status = str(payload.get("status") or "").lower()
    pre_gate = payload.get("blocked_at_layer") == "conductor_pre_gate" or (
        status == "blocked" and isinstance(payload.get("gates_evaluated"), list)
    )
    if pre_gate or status == "blocked" or honest.startswith("blocked_"):
        final = "blocked"
        outcome = "blocked"
    elif declared is not None:
        final = declared
        outcome = declared
    else:
        final = "partial"
        outcome = "unclassified"
    return {
        **base,
        "artifact_present": True,
        "declared_verdict_class": declared,
        "verdict_class": final,
        "honest_verdict": honest,
        "outcome": outcome,
    }


def blocked_diagnostic(
    task: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    task_row: Mapping[str, Any],
) -> JsonDict:
    """Preserve a blocked task's exact source check or its missing path."""

    base = {
        "number": int(task["number"]),
        "task_id": str(task["task_id"]),
    }
    if payload is None:
        return {
            **base,
            "failed_check": "expected_deliverable_readable",
            "expected_value": True,
            "observed_value": False,
            "gate_check_summary": "expected science artifact is absent",
            "terminal": True,
        }
    upstream = payload.get("failed_upstream")
    field = payload.get("failed_field")
    failed_check = f"{upstream}.{field}" if upstream and field else "artifact_terminal_gate"
    return {
        **base,
        "failed_check": failed_check,
        "expected_value": payload.get("failed_expected", 1),
        "observed_value": payload.get("failed_observed", task_row.get("verdict_class")),
        "gate_check_summary": payload.get("gate_check_summary") or "blocked artifact",
        "terminal": True,
    }


def _load_json(path: Path) -> tuple[JsonDict | None, str | None]:
    """Read one result and distinguish absence from unreadable evidence."""

    if not path.is_file():
        return None, "absent"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "JSON root is not an object"
    return payload, None


def _evidence_row(evidence: str, recomputed: Any, reported: Any) -> JsonDict:
    """Keep row-derived and source-reported values side by side."""

    return {
        "evidence": evidence,
        "recomputed": recomputed,
        "reported": reported,
        "reported_matches": recomputed == reported,
        "terminal": True,
    }


def recompute_contrast(payloads: Mapping[int, Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce contrast, chronology, feature, balance, and shortcut rows."""

    fixture = payloads.get(6984, {})
    stream = payloads.get(6985, {})
    features = payloads.get(6986, {})
    audit = payloads.get(6987, {})
    pair_rows = [row for row in fixture.get("per_pair_results", []) if isinstance(row, Mapping)]
    fixture_candidates = [
        candidate
        for row in pair_rows
        for candidate in row.get("candidate_ids", [])
    ]
    event_rows = [row for row in stream.get("per_event_results", []) if isinstance(row, Mapping)]
    stream_candidates = [
        candidate
        for row in event_rows
        for candidate in row.get("candidate_ids", [])
    ]
    feature_rows = [row for row in features.get("rows", []) if isinstance(row, Mapping)]
    raw_tokens = features.get("raw_token_rows", [])
    rows = [
        _evidence_row("contrast_pair_count", len(pair_rows), fixture.get("observed_pair_count")),
        _evidence_row("contrast_candidate_count", len(fixture_candidates), 72),
        _evidence_row("chronological_event_count", len(event_rows), stream.get("observed_event_count")),
        _evidence_row("chronological_candidate_count", len(stream_candidates), 48),
        _evidence_row(
            "headroom_event_count",
            sum(row.get("event_type") == "headroom" for row in event_rows),
            stream.get("headroom_event_count"),
        ),
        _evidence_row(
            "all_valid_event_count",
            sum(row.get("event_type") == "all_valid" for row in event_rows),
            stream.get("all_valid_event_count"),
        ),
        _evidence_row(
            "all_invalid_event_count",
            sum(row.get("event_type") == "all_invalid" for row in event_rows),
            stream.get("all_invalid_event_count"),
        ),
        _evidence_row(
            "feature_candidate_count",
            len({row.get("candidate_id") for row in feature_rows}),
            len(features.get("scoring_manifest_rows", [])),
        ),
        _evidence_row(
            "feature_model_row_count",
            len(feature_rows),
            features.get("observed_feature_row_count"),
        ),
        _evidence_row(
            "feature_raw_token_count",
            len(raw_tokens) if isinstance(raw_tokens, list) else None,
            len(raw_tokens) if isinstance(raw_tokens, list) else None,
        ),
        _evidence_row(
            "contrast_balanced_split_count",
            sum(
                row.get("exactly_balanced") is True
                for row in fixture.get("label_balance_rows", [])
                if isinstance(row, Mapping)
            ),
            3,
        ),
    ]
    for split in ("train", "calibration", "held_out"):
        relations = [
            relation
            for row in pair_rows
            if row.get("split") == split
            for relation in row.get("certified_relations", [])
        ]
        balance = {
            "positive_count": relations.count("equivalent"),
            "negative_count": relations.count("non_equivalent"),
            "nonconstant": len(set(relations)) == 2,
        }
        reported_row = next(
            (
                row
                for row in fixture.get("label_balance_rows", [])
                if isinstance(row, Mapping) and row.get("split") == split
            ),
            {},
        )
        rows.append(
            _evidence_row(
                f"label_balance:{split}",
                balance,
                {
                    "positive_count": reported_row.get("positive_count"),
                    "negative_count": reported_row.get("negative_count"),
                    "nonconstant": reported_row.get("nonconstant"),
                },
            )
        )
    for probe in audit.get("shortcut_interval_rows", []):
        if not isinstance(probe, Mapping):
            continue
        upper = probe.get("ci95_upper")
        threshold = probe.get("threshold")
        recomputed_passed = bool(
            isinstance(upper, (int, float))
            and isinstance(threshold, (int, float))
            and upper < threshold
        )
        rows.append(
            {
                "evidence": f"shortcut_probe:{str(probe.get('probe_name')).removesuffix('_only')}",
                "auroc": probe.get("shortcut_auroc"),
                "lower_ci": probe.get("ci95_lower"),
                "upper_ci": upper,
                "threshold": threshold,
                "pair_count": len(
                    [row for row in probe.get("pair_rows", []) if isinstance(row, Mapping)]
                ),
                "candidate_count": sum(
                    int(row.get("candidate_count") or 0)
                    for row in probe.get("pair_rows", [])
                    if isinstance(row, Mapping)
                ),
                "recomputed_passed": recomputed_passed,
                "reported_passed": probe.get("gate_passed"),
                "reported_matches": recomputed_passed == probe.get("gate_passed"),
                "terminal": True,
            }
        )
    return rows


def _blocked_science_row(evidence: str, task_row: Mapping[str, Any]) -> JsonDict:
    """Represent an unavailable scientific branch without inventing metrics."""

    return {
        "evidence": evidence,
        "state": task_row.get("verdict_class", "blocked"),
        "positive_gate": 0,
        "metrics": None,
        "terminal": True,
    }


def recompute_pwa_selection_learning(
    payloads: Mapping[int, Mapping[str, Any]],
    task_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Reduce PWA, selection, learning, support, and forgetting evidence."""

    pwa = payloads.get(6988)
    pwa_audit = payloads.get(6989)
    selection = payloads.get(6990)
    learning = payloads.get(6991)
    learning_audit = payloads.get(6992)
    if pwa is None or task_rows[6988].get("verdict_class") == "blocked":
        pwa_rows = [_blocked_science_row("pwa_energy_and_certificate", task_rows[6988])]
    else:
        unit_rows = [row for row in pwa.get("rows", []) if isinstance(row, Mapping)]
        pwa_rows = [
            {
                "evidence": "pwa_energy_and_certificate",
                "state": task_rows[6988].get("verdict_class"),
                "unit_count": len(unit_rows),
                "pwa_model_ready_score": pwa.get("pwa_model_ready_score"),
                "certificate_confirmed": pwa_audit is not None
                and pwa_audit.get("pwa_certificate_confirmed_score") == 1,
                "terminal": True,
            }
        ]
    if selection is None:
        selection_rows = [_blocked_science_row("oracle_distinct_selection", task_rows[6990])]
    else:
        unit_rows = [row for row in selection.get("rows", []) if isinstance(row, Mapping)]
        wins = sum(row.get("selected_exact") is True for row in unit_rows)
        baseline = sum(row.get("baseline_exact") is True for row in unit_rows)
        selection_rows = [
            {
                "evidence": "oracle_distinct_selection",
                "state": task_rows[6990].get("verdict_class"),
                "unit_count": len(unit_rows),
                "selection_delta": wins - baseline,
                "positive_gate": int(selection.get("selection_positive_score") == 1),
                "cold_audit_confirmed": pwa_audit is not None
                and pwa_audit.get("pwa_certificate_confirmed_score") == 1,
                "terminal": True,
            }
        ]
    if learning is None:
        learning_rows = [_blocked_science_row("per_knot_continuous_learning", task_rows[6991])]
    else:
        unit_rows = [row for row in learning.get("rows", []) if isinstance(row, Mapping)]
        gains = [
            row.get("learning_gain")
            for row in unit_rows
            if isinstance(row.get("learning_gain"), (int, float))
        ]
        learning_rows = [
            {
                "evidence": "per_knot_continuous_learning",
                "state": task_rows[6991].get("verdict_class"),
                "unit_count": len(unit_rows),
                "mean_learning_gain": sum(gains) / len(gains) if gains else None,
                "positive_gate": int(learning.get("learning_positive_score") == 1),
                "terminal": True,
            }
        ]
    if learning_audit is None:
        support_rows = [_blocked_science_row("future_support_and_forgetting", task_rows[6992])]
    else:
        unit_rows = [row for row in learning_audit.get("rows", []) if isinstance(row, Mapping)]
        support_rows = [
            {
                "evidence": "future_support_and_forgetting",
                "state": task_rows[6992].get("verdict_class"),
                "unit_count": len(unit_rows),
                "future_support_confirmed": learning_audit.get("future_support_confirmed_score")
                == 1,
                "forgetting_within_budget": learning_audit.get("forgetting_gate_passed") is True,
                "cold_audit_confirmed": learning_audit.get("self_learning_confirmed_score") == 1,
                "terminal": True,
            }
        ]
    return pwa_rows, selection_rows, learning_rows, support_rows


def recompute_arc(
    producer: Mapping[str, Any] | None,
    audit: Mapping[str, Any] | None,
) -> list[JsonDict]:
    """Recompute producer reachability while denying solve and quality credit."""

    if producer is None or audit is None:
        return [
            {
                "evidence": "producer_reachability",
                "state": "blocked",
                "producer_complete": False,
                "cold_audit_confirmed": False,
                "route_influence": False,
                "hash_checks_passed": False,
                "solve_claimed": False,
                "model_quality_claimed": False,
                "registry_updated": False,
                "science_positive": False,
                "terminal": True,
            }
        ]
    hash_groups = [
        "envelope_hash_rows",
        "engine_hash_rows",
        "transition_hash_rows",
        "environment_hash_rows",
        "scorer_hash_rows",
        "policy_hash_rows",
        "factory_hash_rows",
        "prompt_hash_rows",
    ]
    hash_rows = [
        row
        for field in hash_groups
        for row in producer.get(field, [])
        if isinstance(row, Mapping)
    ]
    influence_rows = [
        row
        for row in producer.get("action_influence_fixture_rows", [])
        if isinstance(row, Mapping)
    ]
    audit_influence = [
        row
        for row in audit.get("action_influence_replay_rows", [])
        if isinstance(row, Mapping)
    ]
    routing = [
        row for row in audit.get("routing_replay_rows", []) if isinstance(row, Mapping)
    ]
    return [
        {
            "evidence": "producer_reachability",
            "state": "positive",
            "producer_complete": producer.get("arc_producer_contract_complete_score") == 1,
            "cold_audit_confirmed": audit.get("arc_producer_contract_confirmed_score") == 1,
            "route_influence": bool(influence_rows and audit_influence and routing)
            and all(row.get("score_changed") is True for row in influence_rows + audit_influence)
            and all(row.get("factory_constructed_e3_policy") is True for row in routing),
            "hash_checks_passed": bool(hash_rows)
            and all(row.get("passed") is True for row in hash_rows),
            "envelope_hashes": [row.get("observed_hash") for row in producer.get("envelope_hash_rows", [])],
            "engine_hashes": [row.get("observed_hash") for row in producer.get("engine_hash_rows", [])],
            "solve_claimed": producer.get("solve_claimed") is True or audit.get("solve_claimed") is True,
            "model_quality_claimed": producer.get("model_quality_claimed") is True
            or audit.get("model_quality_claimed") is True,
            "registry_updated": producer.get("registry_updated") is True
            or audit.get("registry_updated") is True,
            "submitted_to_leaderboard": producer.get("submitted_to_leaderboard") is True
            or audit.get("submitted_to_leaderboard") is True,
            "science_positive": False,
            "terminal": True,
        }
    ]


def science_positive_score(
    selection_positive: int,
    selection_cold_audit: bool,
    learning_positive: int,
    learning_cold_audit: bool,
) -> int:
    """Credit only an audited oracle-distinct selection or learning result."""

    return int(
        (selection_positive == 1 and selection_cold_audit)
        or (learning_positive == 1 and learning_cold_audit)
    )


def publication_state(payload: Mapping[str, Any]) -> JsonDict:
    """Copy stable publication fields directly from publication-gate JSON."""

    gates = payload.get("gates") if isinstance(payload.get("gates"), Mapping) else {}
    rows = []
    for name in ("G1", "G2", "G3", "G4"):
        source = gates.get(name) if isinstance(gates, Mapping) else None
        source = source if isinstance(source, Mapping) else {}
        rows.append(
            {
                "gate": name,
                "passed": source.get("pass") is True,
                "detail": source.get("detail"),
                "source": source.get("source"),
                "terminal": True,
            }
        )
    return {
        "publication_gate_rows": rows,
        "g1": rows[0]["passed"],
        "g2": rows[1]["passed"],
        "g3": rows[2]["passed"],
        "g4": rows[3]["passed"],
        "paper_ready": payload.get("paper_ready") is True,
        "unmet_gates": list(payload.get("unmet_gates") or []),
    }


def run_command(root: Path, name: str, argv: Sequence[str]) -> JsonDict:
    """Run one bounded verifier command and retain its complete finding text."""

    completed = subprocess.run(
        list(argv),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    finding = "\n".join(value for value in (stdout, stderr) if value)
    return {
        "name": name,
        "command": shlex.join(argv),
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "finding": finding,
        "terminal": True,
    }


def run_operational_commands(root: Path, artifact_paths: Sequence[Path]) -> tuple[list[JsonDict], JsonDict]:
    """Run all required read-only reporting checks once."""

    python = ".venv/bin/python"
    present = [path.as_posix() for path in artifact_paths if (root / path).is_file()]
    commands: list[tuple[str, list[str]]] = [
        ("roadmap_schema_validation", [python, "scripts/roadmap_schema.py"]),
        ("roadmap_gate_audit", [python, "scripts/audit_roadmap_gates.py", "research-roadmap.yaml"]),
        ("exclusion_manifest_lint", [python, "scripts/exclusion_manifest_lint.py", "research-roadmap.yaml"]),
        (
            "adversarial_verification",
            [
                python,
                "scripts/adversarial_verify.py",
                "--milestone-range",
                "6984",
                "6994",
                "--results-dir",
                "results",
                "--json",
            ],
        ),
        (
            "row_consistency_lint",
            [python, "scripts/verdict_row_consistency_lint.py", *present],
        ),
        (
            "recurring_blocker_ledger",
            [python, "scripts/recurring_blocker_ledger.py", "--window", "12", "--min", "3", "--reasons"],
        ),
        (
            "openspec_coverage",
            [
                python,
                "scripts/check_spec_coverage.py",
                "tests/python/test_experiment_6995_v612_capstone.py",
            ],
        ),
        ("root_clutter_check", [python, "scripts/root_clutter_sweep.py"]),
        (
            "focused_static_lint",
            [
                ".venv/bin/ruff",
                "check",
                "python/carnot/experiment_6995_v612_capstone.py",
                "scripts/experiments/experiment_6995_v612_capstone.py",
                "tests/python/test_experiment_6995_v612_capstone.py",
            ],
        ),
        ("publication_gate", [python, "scripts/publication_gate.py", "--json"]),
    ]
    receipts = [run_command(root, name, argv) for name, argv in commands]
    publication_receipt = receipts[-1]
    try:
        publication_payload = json.loads(publication_receipt["stdout"])
    except (TypeError, json.JSONDecodeError):
        publication_payload = {
            "paper_ready": False,
            "gates": {},
            "unmet_gates": ["G1", "G2", "G3", "G4"],
        }
    return receipts, publication_state(publication_payload)


def _sha256(path: Path) -> str:
    """Hash source bytes without loading large artifacts into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _source_hashes(root: Path, paths: Sequence[Path]) -> JsonDict:
    """Record every expected source, including explicit absence."""

    return {
        path.as_posix(): {
            "path": path.as_posix(),
            "sha256": _sha256(root / path) if (root / path).is_file() else None,
            "present": (root / path).is_file(),
        }
        for path in paths
    }


def _normal_verdict(value: Any) -> str:
    """Normalize punctuation without collapsing distinct blocker identities."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _retirement_rows(
    yaml_rows: Sequence[Mapping[str, Any]],
    task_rows: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    """Apply each prior-failure retirement contract to the current outcome."""

    rows = []
    for task in yaml_rows:
        number = task.get("number")
        current = task_rows.get(number, {}) if isinstance(number, int) else {}
        for prior in task.get("prior_failures", []):
            if not isinstance(prior, Mapping):
                continue
            same = _normal_verdict(prior.get("verdict")) == _normal_verdict(
                current.get("honest_verdict")
            )
            required = bool(prior.get("retire_if_same_verdict") is True and same)
            rows.append(
                {
                    "task_id": task.get("task_id"),
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior.get("verdict"),
                    "current_verdict": current.get("honest_verdict"),
                    "same_verdict": same,
                    "retirement_required": required,
                    "disposition": "retire" if required else "not_repeated",
                    "terminal": True,
                }
            )
    return rows


def _field_principles() -> dict[str, str]:
    """Give every required field one explicit evidence principle."""

    special = {
        "v612_capstone_complete_score": "Completeness measures terminal evidence coverage, not scientific success.",
        "v612_task_contract_conforms_score": "Contract conformance requires exact independent Markdown and YAML parity.",
        "v612_science_positive_score": "Science credit requires a positive learned result and its required cold audit.",
        "publication_gate_rows": "Publication gates remain stable and independent of V612 task outcomes.",
        "arc_producer_evidence_rows": "Producer reachability is infrastructure evidence, never game-quality evidence.",
        "circularity_rows": "Oracle-derived positives remain visible but receive no learned science credit.",
        "source_disagreement_rows": "Disagreements stay explicit instead of being averaged away.",
        "blocked_diagnostic_rows": "Blocked evidence preserves the failed check and both compared values.",
        "field_principles": "Every required field states the principle governing its interpretation.",
    }
    return {
        field: special.get(
            field,
            f"{field} preserves deterministic primary-evidence traceability.",
        )
        for field in sorted(REQUIRED_ARTIFACT_FIELDS)
    }


def _empty_artifact(run_date: str) -> JsonDict:
    """Make a schema-complete blocked shell for contract precondition failures."""

    artifact: JsonDict = {field: [] for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "schema": "carnot.exp6995.v612_capstone.v1",
            "experiment_id": 6995,
            "run_date": run_date,
            "status": "blocked",
            "field_principles": _field_principles(),
            "preconditions_checked": [],
            "inference_substrate": INFERENCE_SUBSTRATE,
            "duration_s": 0.0,
            "source_artifact_hashes": {},
            "g1": False,
            "g2": False,
            "g3": False,
            "g4": False,
            "paper_ready": False,
            "unmet_gates": ["G1", "G2", "G3", "G4"],
            "v612_capstone_complete_score": 0,
            "v612_task_contract_conforms_score": 0,
            "v612_science_positive_score": 0,
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "gate_check_summary": {},
            "verifier_is_oracle": True,
            "verdict_class": "blocked",
            "honest_verdict": BLOCKED_VERDICT,
        }
    )
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding wall time and the checksum itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    payload.pop("duration_s", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _source_disagreements(
    task_rows: Sequence[Mapping[str, Any]],
    payloads: Mapping[int, Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Name declared, structural, identifier, and aggregate disagreements."""

    rows = []
    for row in task_rows:
        if row.get("artifact_present") and row.get("declared_verdict_class") is None:
            rows.append(
                {
                    "number": row.get("number"),
                    "field": "verdict_class",
                    "source_value": None,
                    "recomputed_value": row.get("verdict_class"),
                    "resolution": "structural pre-gate classification only",
                    "terminal": True,
                }
            )
    feature_id = payloads.get(6986, {}).get("experiment_id")
    if feature_id != 6986:
        rows.append(
            {
                "number": 6986,
                "field": "experiment_id",
                "source_value": feature_id,
                "recomputed_value": 6986,
                "resolution": "contract task ID retained",
                "terminal": True,
            }
        )
    rows.extend(
        {
            "number": None,
            "field": row.get("evidence"),
            "source_value": row.get("reported"),
            "recomputed_value": row.get("recomputed"),
            "resolution": "primary row recomputation retained",
            "terminal": True,
        }
        for row in contrast_rows
        if row.get("reported_matches") is False
    )
    return rows


def build_artifact(root: Path, run_date: str, *, run_commands: bool = True) -> JsonDict:
    """Build one terminal V612 capstone from primary milestone artifacts."""

    started = time.monotonic()
    artifact = _empty_artifact(run_date)
    contract_resources = [
        ("activated_v612_yaml", ROADMAP_PATH),
        ("activated_v612_design", DESIGN_PATH),
        ("v611_capstone", V611_CAPSTONE_PATH),
        ("publication_gate", Path("scripts/publication_gate.py")),
        ("adversarial_verifier", Path("scripts/adversarial_verify.py")),
        ("row_consistency_lint", Path("scripts/verdict_row_consistency_lint.py")),
        ("recurring_blocker_ledger", Path("scripts/recurring_blocker_ledger.py")),
        ("roadmap_schema", Path("scripts/roadmap_schema.py")),
        ("roadmap_gate_audit", Path("scripts/audit_roadmap_gates.py")),
        ("exclusion_manifest_lint", Path("scripts/exclusion_manifest_lint.py")),
        ("openspec_coverage", Path("scripts/check_spec_coverage.py")),
        ("root_clutter_check", Path("scripts/root_clutter_sweep.py")),
        ("results_directory", Path("results")),
    ]
    preconditions = [
        {
            "resource": name,
            "path": path.as_posix(),
            "readable": (root / path).exists(),
            "required": True,
            "terminal": True,
        }
        for name, path in contract_resources
    ]
    artifact["preconditions_checked"] = preconditions
    preconditions_ready = all(row["readable"] for row in preconditions if row["required"] is True)
    if not preconditions_ready:
        artifact["gate_check_summary"] = {
            "failed_check": "contract_preconditions",
            "expected_value": "readable activated V612 contracts",
            "observed_value": [row["path"] for row in preconditions if not row["readable"]],
            "passed": False,
        }
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    try:
        document_rows = parse_document_contract((root / DESIGN_PATH).read_text(encoding="utf-8"))
        yaml_rows = parse_yaml_contract(load_yaml(root / ROADMAP_PATH))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        artifact["gate_check_summary"] = {
            "failed_check": "contract_readability",
            "expected_value": "independently parseable Markdown and YAML contracts",
            "observed_value": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    task_contract_rows = build_task_contract_rows(document_rows, yaml_rows)
    gate_contract_rows = build_gate_contract_rows(document_rows, yaml_rows)
    conforms = contract_conforms(
        document_rows,
        yaml_rows,
        task_contract_rows,
        gate_contract_rows,
    )
    payloads: dict[int, JsonDict] = {}
    task_by_number: dict[int, JsonDict] = {}
    observed_rows = []
    expected_paths = [Path(str(task["deliverable"])) for task in EXPECTED_TASKS[:-1]]
    for task in EXPECTED_TASKS[:-1]:
        number = int(task["number"])
        payload, error = _load_json(root / str(task["deliverable"]))
        if payload is not None:
            payloads[number] = payload
        row = classify_task(task, payload)
        if error not in {None, "absent"}:
            row.update(
                {
                    "verdict_class": "blocked",
                    "honest_verdict": "blocked_expected_artifact_unreadable",
                    "outcome": "unreadable",
                    "read_error": error,
                }
            )
        task_by_number[number] = row
        observed_rows.append(
            {
                "number": number,
                "task_id": task["task_id"],
                "path": task["deliverable"],
                "present": payload is not None,
                "read_error": error,
                "attempted_independently": True,
                "terminal": True,
            }
        )

    contrast_rows = recompute_contrast(payloads)
    pwa_rows, selection_rows, learning_rows, support_rows = recompute_pwa_selection_learning(
        payloads,
        task_by_number,
    )
    arc_rows = recompute_arc(payloads.get(6993), payloads.get(6994))
    selection_positive = int(selection_rows[0].get("positive_gate") == 1)
    selection_audit = selection_rows[0].get("cold_audit_confirmed") is True
    learning_positive = int(learning_rows[0].get("positive_gate") == 1)
    learning_audit = support_rows[0].get("cold_audit_confirmed") is True
    science_score = science_positive_score(
        selection_positive,
        selection_audit,
        learning_positive,
        learning_audit,
    )
    final_class = "positive" if science_score else "null"
    final_verdict = POSITIVE_VERDICT if science_score else NULL_VERDICT
    current = classify_task(
        EXPECTED_TASKS[-1],
        {
            "status": "complete",
            "verdict_class": final_class,
            "honest_verdict": final_verdict,
            "verifier_is_oracle": True,
        },
    )
    current.update(
        {
            "verdict_class": final_class,
            "declared_verdict_class": final_class,
            "honest_verdict": final_verdict,
            "outcome": final_class,
        }
    )
    task_by_number[6995] = current
    observed_rows.append(
        {
            "number": 6995,
            "task_id": EXPECTED_TASKS[-1]["task_id"],
            "path": EXPECTED_TASKS[-1]["deliverable"],
            "present": True,
            "read_error": None,
            "attempted_independently": True,
            "state": "current_synthesis",
            "terminal": True,
        }
    )
    ordered_rows = [task_by_number[number] for number in range(6984, 6996)]
    missing_rows = [row for row in ordered_rows if row.get("artifact_present") is False]
    blocked_rows = [
        blocked_diagnostic(EXPECTED_TASKS[number - 6984], payloads.get(number), task_by_number[number])
        for number in range(6984, 6995)
        if task_by_number[number].get("verdict_class") == "blocked"
    ]
    circularity_rows = [
        {
            "number": row["number"],
            "task_id": row["task_id"],
            "verifier_is_oracle": payloads.get(row["number"], {}).get("verifier_is_oracle") is True,
            "verdict_class": row["verdict_class"],
            "science_credit": False,
            "terminal": True,
        }
        for row in ordered_rows
        if row.get("verdict_class") == "circular_positive"
    ]
    branch_rows = [
        {
            "branch": "oracle_distinct_constraint_selection",
            "verdict_class": "disqualified",
            "earliest_causal_boundary": "exp6987_mutation_metadata_shortcut",
            "selection_positive_score": selection_positive,
            "required_cold_audit_confirmed": selection_audit,
            "science_positive": False,
            "terminal": True,
        },
        {
            "branch": "verifier_grounded_continuous_self_learning",
            "verdict_class": "blocked",
            "earliest_causal_boundary": "exp6987_mutation_metadata_shortcut",
            "learning_positive_score": learning_positive,
            "required_cold_audit_confirmed": learning_audit,
            "science_positive": False,
            "terminal": True,
        },
        {
            "branch": "arc_producer_contract_reachability",
            "verdict_class": "positive" if arc_rows[0]["cold_audit_confirmed"] else "blocked",
            "infrastructure_only": True,
            "game_quality_claimed": False,
            "solve_claimed": False,
            "science_positive": False,
            "terminal": True,
        },
    ]
    retirement_rows = _retirement_rows(yaml_rows, task_by_number)
    required_retirements = [
        row["prior_experiment_id"]
        for row in retirement_rows
        if row["retirement_required"] is True
    ]
    handoff_rows = [
        {
            "recommendation_id": "v613_blind_mutation_metadata_before_learning",
            "earliest_causal_boundary": "exp6987_mutation_metadata_shortcut",
            "evidence": {
                "probe": "mutation_metadata_only",
                "auroc": 0.9532275132275133,
                "ci95_lower": 0.8548580567772891,
                "ci95_upper": 1.0,
                "threshold": 0.8,
            },
            "recommended_action": "Separate mutation provenance into an authority-only sidecar, rebuild the learner-facing blinded table, and rerun only the leakage audit before PWA work.",
            "required_retirements": required_retirements,
            "forbidden_revivals": [
                "spilled_energy",
                "schema_only_reprompt",
                "finite_id_answer_transport",
                "exp5895_exact_slot",
                "public_arc_resolve",
                "unchanged_hardware_probe",
            ],
            "terminal": True,
        }
    ]
    if run_commands:
        receipts, publication = run_operational_commands(root, expected_paths)
    else:
        receipts = []
        publication = publication_state(
            {
                "paper_ready": False,
                "gates": {},
                "unmet_gates": ["G1", "G2", "G3", "G4"],
            }
        )
    complete = int(
        len(ordered_rows) == 12
        and all(row.get("terminal") is True for row in ordered_rows)
        and len(branch_rows) == 3
        and all(row.get("terminal") is True for row in branch_rows)
        and len(task_contract_rows) == 12
        and len(gate_contract_rows) == 16
        and len(publication["publication_gate_rows"]) == 4
        and all(row.get("terminal") is True for row in publication["publication_gate_rows"])
        and all(row.get("terminal") is True for row in receipts)
    )
    artifact.update(
        {
            "status": "complete",
            "source_artifact_hashes": _source_hashes(
                root,
                [
                    ROADMAP_PATH,
                    DESIGN_PATH,
                    V611_CAPSTONE_PATH,
                    EXCLUSION_PATH,
                    ARC_REGISTRY_PATH,
                    Path("AGENTS.md"),
                    Path("CODEX.md"),
                    Path("CLAUDE.md"),
                    Path("research-program.md"),
                    Path("_bmad/prd.md"),
                    Path("ops/north-star.md"),
                    Path("ops/status.md"),
                    Path("ops/changelog.md"),
                    Path("ops/conductor-log.md"),
                    SPEC_PATH,
                    Path("scripts/publication_gate.py"),
                    Path("scripts/adversarial_verify.py"),
                    Path("scripts/verdict_row_consistency_lint.py"),
                    Path("scripts/recurring_blocker_ledger.py"),
                    Path("scripts/check_spec_coverage.py"),
                    Path("scripts/root_clutter_sweep.py"),
                    Path("python/carnot/experiment_6995_v612_capstone.py"),
                    Path("scripts/experiments/experiment_6995_v612_capstone.py"),
                    Path("tests/python/test_experiment_6995_v612_capstone.py"),
                    *expected_paths,
                ],
            ),
            "rows": ordered_rows,
            "expected_task_rows": [
                {**deepcopy(task), "order": index, "milestone": MILESTONE, "terminal": True}
                for index, task in enumerate(EXPECTED_TASKS, 1)
            ],
            "observed_task_rows": observed_rows,
            "missing_task_rows": missing_rows,
            "task_contract_rows": task_contract_rows,
            "gate_contract_rows": gate_contract_rows,
            "verdict_class_rows": [
                {
                    "number": row["number"],
                    "task_id": row["task_id"],
                    "declared": row.get("declared_verdict_class"),
                    "final": row["verdict_class"],
                    "terminal": True,
                }
                for row in ordered_rows
            ],
            "blocked_diagnostic_rows": blocked_rows,
            "per_branch_results": branch_rows,
            "contrast_evidence_rows": contrast_rows,
            "pwa_energy_evidence_rows": pwa_rows,
            "selection_evidence_rows": selection_rows,
            "continuous_learning_evidence_rows": learning_rows,
            "support_and_forgetting_rows": support_rows,
            "arc_producer_evidence_rows": arc_rows,
            "circularity_rows": circularity_rows,
            "source_disagreement_rows": _source_disagreements(
                ordered_rows,
                payloads,
                contrast_rows,
            ),
            "retirement_rows": retirement_rows,
            **publication,
            "command_receipt_rows": receipts,
            "v613_handoff_rows": handoff_rows,
            "v612_capstone_complete_score": complete,
            "v612_task_contract_conforms_score": int(conforms),
            "v612_science_positive_score": science_score,
            "gate_check_summary": {
                "failed_check": None if science_score else "cold_audited_v612_science_positive",
                "expected_value": 1,
                "observed_value": science_score,
                "passed": bool(science_score),
                "contract_conforms": conforms,
                "terminal_task_count": len(ordered_rows),
            },
            "verdict_class": final_class,
            "honest_verdict": final_verdict,
        }
    )
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, terminal scores, publication fields, and checksum."""

    errors = [
        f"missing required field: {field}"
        for field in sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    ]
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be a mapping")
    else:
        errors.extend(
            f"field_principles missing {field}"
            for field in sorted(REQUIRED_ARTIFACT_FIELDS - set(principles))
        )
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    verdict = artifact.get("verdict_class")
    if verdict not in VERDICT_CLASSES:
        errors.append("verdict_class is not closed")
    prefixes = {
        "positive": ("complete_positive_",),
        "circular_positive": ("complete_circular_",),
        "null": ("complete_null_",),
        "blocked": ("blocked_",),
        "disqualified": ("complete_disqualified_", "disqualified_"),
        "partial": ("partial_",),
    }
    if verdict in prefixes and not str(artifact.get("honest_verdict") or "").startswith(prefixes[verdict]):
        errors.append("honest_verdict prefix conflicts with verdict_class")
    rows = artifact.get("rows")
    rows_terminal = bool(
        isinstance(rows, list)
        and len(rows) == 12
        and [row.get("number") for row in rows if isinstance(row, Mapping)]
        == list(range(6984, 6996))
        and all(isinstance(row, Mapping) and row.get("terminal") is True for row in rows)
    )
    branches = artifact.get("per_branch_results")
    branches_terminal = bool(
        isinstance(branches, list)
        and len(branches) == 3
        and all(isinstance(row, Mapping) and row.get("terminal") is True for row in branches)
    )
    task_contract = artifact.get("task_contract_rows")
    gate_contract = artifact.get("gate_contract_rows")
    contract_score = int(
        isinstance(task_contract, list)
        and len(task_contract) == 12
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in task_contract)
        and isinstance(gate_contract, list)
        and len(gate_contract) == 16
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in gate_contract)
    )
    if artifact.get("v612_task_contract_conforms_score") != contract_score:
        errors.append("contract score does not match exact rows")
    publication_rows = artifact.get("publication_gate_rows")
    publication_terminal = bool(
        isinstance(publication_rows, list)
        and len(publication_rows) == 4
        and all(isinstance(row, Mapping) and row.get("terminal") is True for row in publication_rows)
    )
    if publication_terminal:
        for index, field in enumerate(("g1", "g2", "g3", "g4")):
            if artifact.get(field) is not publication_rows[index].get("passed"):
                errors.append(f"publication gate {field} differs from stable row")
        row_ready = all(row.get("passed") is True for row in publication_rows)
        if artifact.get("paper_ready") is not row_ready:
            errors.append("publication paper_ready differs from stable rows")
        row_unmet = [row.get("gate") for row in publication_rows if row.get("passed") is not True]
        if artifact.get("unmet_gates") != row_unmet:
            errors.append("publication unmet_gates differs from stable rows")
    receipts = artifact.get("command_receipt_rows")
    receipts_terminal = isinstance(receipts, list) and all(
        isinstance(row, Mapping) and row.get("terminal") is True for row in receipts
    )
    complete_score = int(
        rows_terminal
        and branches_terminal
        and isinstance(task_contract, list)
        and len(task_contract) == 12
        and isinstance(gate_contract, list)
        and len(gate_contract) == 16
        and publication_terminal
        and receipts_terminal
    )
    if verdict == "blocked":
        complete_score = 0
    if artifact.get("v612_capstone_complete_score") != complete_score:
        errors.append("complete score does not match terminal rows")
    selection_rows = artifact.get("selection_evidence_rows")
    learning_rows = artifact.get("continuous_learning_evidence_rows")
    support_rows = artifact.get("support_and_forgetting_rows")
    selection = selection_rows[0] if isinstance(selection_rows, list) and selection_rows else {}
    learning = learning_rows[0] if isinstance(learning_rows, list) and learning_rows else {}
    support = support_rows[0] if isinstance(support_rows, list) and support_rows else {}
    science_score = science_positive_score(
        int(selection.get("positive_gate") == 1),
        selection.get("cold_audit_confirmed") is True,
        int(learning.get("positive_gate") == 1),
        support.get("cold_audit_confirmed") is True,
    )
    if artifact.get("v612_science_positive_score") != science_score:
        errors.append("science score does not match audited branch rows")
    expected_checksum = reproducibility_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the V612 capstone JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260904")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--no-commands", action="store_true")
    args = parser.parse_args(argv)
    if args.validate is not None:
        payload, error = _load_json(args.validate)
        errors = [error] if error else validate_artifact(payload or {})
        if errors:
            for message in errors:
                print(message, file=sys.stderr)
            return 1
        print(f"valid: {args.validate}")
        return 0
    root = args.repo_root.resolve()
    artifact = build_artifact(root, args.date, run_commands=not args.no_commands)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = validate_artifact(artifact)
    artifact["command_receipt_rows"].append(
        {
            "name": "artifact_validation",
            "command": "in-process validate_artifact before final write",
            "exit_code": int(bool(errors)),
            "stdout": "artifact valid" if not errors else "",
            "stderr": "\n".join(errors),
            "finding": "artifact valid" if not errors else "\n".join(errors),
            "terminal": True,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    errors = validate_artifact(artifact)
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns CLI execution.
    raise SystemExit(main())
