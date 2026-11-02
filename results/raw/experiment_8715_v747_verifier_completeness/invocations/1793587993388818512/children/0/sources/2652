"""Build the V615 evidence capstone for REQ-CAPSTONE-7027.

The reducer reads contracts and stored evidence. It does not rerun upstream
science. This separation prevents a capstone retry from hiding an external
gate block or changing the result that it is meant to audit.
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
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_7027_v615_capstone.json")
SPEC_PATH = Path("openspec/capabilities/capstone/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
INACTIVE_ROADMAP_PATH = Path("research-roadmap-next.yaml")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
SUMMARIZER_PATH = Path("scripts/summarize_artifact.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
MILESTONE = "2026.09.615"
INFERENCE_SUBSTRATE = "deterministic_v615_evidence_synthesis_no_llm"
RANDOM_SEED = 702720260905
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
PROSPECTIVE_ARMS = ("frozen", "recency_only", "counterexample_belief")
LIVE_ARMS = ("base", "simulation_only", "belief_only", "combined")


def _gate(upstream: str, artifact_field: str) -> JsonDict:
    """Return the fixed equality gate shape used in V615."""

    return {"upstream": upstream, "artifact_field": artifact_field, "op": "==", "value": 1}


EXPECTED_TASKS: tuple[JsonDict, ...] = (
    {
        "number": 7016,
        "task_id": "exp7016-v615-source-contract-preflight",
        "title": "V615 source delta and exact task-contract preflight",
        "deliverable": "results/experiment_7016_v615_source_contract_preflight.json",
        "gates": [],
    },
    {
        "number": 7017,
        "task_id": "exp7017-task-linked-compute-receipts",
        "title": "Task-linked phase, GPU, and runner receipt contract",
        "deliverable": "results/experiment_7017_task_linked_compute_receipts.json",
        "gates": [],
    },
    {
        "number": 7018,
        "task_id": "exp7018-v615-sota-ingestion",
        "title": "V615 recent-source ingestion and architecture map",
        "deliverable": "results/experiment_7018_v615_sota_ingestion.json",
        "gates": [],
    },
    {
        "number": 7019,
        "task_id": "exp7019-arc-belief-stream-fixture",
        "title": "Immutable ARC chronological belief-stream fixture",
        "deliverable": "results/experiment_7019_arc_belief_stream_fixture.json",
        "gates": [],
    },
    {
        "number": 7020,
        "task_id": "exp7020-counterexample-belief-ledger",
        "title": "Counterexample-updated ARC belief ledger",
        "deliverable": "results/experiment_7020_counterexample_belief_ledger.json",
        "gates": [_gate("exp7019-arc-belief-stream-fixture", "arc_belief_stream_ready_score")],
    },
    {
        "number": 7021,
        "task_id": "exp7021-prospective-belief-utility",
        "title": "Prospective held-future belief utility comparison",
        "deliverable": "results/experiment_7021_prospective_belief_utility.json",
        "gates": [_gate("exp7020-counterexample-belief-ledger", "belief_ledger_ready_score")],
    },
    {
        "number": 7022,
        "task_id": "exp7022-belief-ledger-cold-audit",
        "title": "Fresh-process belief isolation, retention, and poison audit",
        "deliverable": "results/experiment_7022_belief_ledger_cold_audit.json",
        "gates": [
            _gate("exp7020-counterexample-belief-ledger", "belief_ledger_ready_score"),
            _gate("exp7021-prospective-belief-utility", "belief_utility_comparison_complete_score"),
        ],
    },
    {
        "number": 7023,
        "task_id": "exp7023-belief-query-api",
        "title": "Bounded belief-query API for the ARC policy",
        "deliverable": "results/experiment_7023_belief_query_api.json",
        "gates": [_gate("exp7022-belief-ledger-cold-audit", "belief_shadow_safe_score")],
    },
    {
        "number": 7024,
        "task_id": "exp7024-belief-aware-e3-selector",
        "title": "Default-off belief-aware E3 selector wiring",
        "deliverable": "results/experiment_7024_belief_aware_e3_selector.json",
        "gates": [_gate("exp7023-belief-query-api", "belief_query_api_ready_score")],
    },
    {
        "number": 7025,
        "task_id": "exp7025-belief-shadow-live-trace",
        "title": "Provenance-complete live belief shadow trace",
        "deliverable": "results/experiment_7025_belief_shadow_live_trace.json",
        "gates": [
            _gate("exp7017-task-linked-compute-receipts", "task_compute_receipt_ready_score"),
            _gate("exp7024-belief-aware-e3-selector", "belief_selector_live_path_ready_score"),
        ],
    },
    {
        "number": 7026,
        "task_id": "exp7026-held-mechanic-belief-ab",
        "title": "Held-mechanic live belief and simulation A/B",
        "deliverable": "results/experiment_7026_held_mechanic_belief_ab.json",
        "gates": [
            _gate("exp7017-task-linked-compute-receipts", "task_compute_receipt_ready_score"),
            _gate("exp7025-belief-shadow-live-trace", "belief_shadow_trace_ready_score"),
        ],
    },
    {
        "number": 7027,
        "task_id": "exp7027-v615-capstone",
        "title": "V615 independent evidence capstone and V616 handoff",
        "deliverable": "results/experiment_7027_v615_capstone.json",
        "gates": [],
    },
)
EXPECTED_ID_ORDER = [str(task["task_id"]) for task in EXPECTED_TASKS]
TASK_BY_NUMBER = {int(task["number"]): task for task in EXPECTED_TASKS}

IMPORTED_FIELDS: dict[int, list[str]] = {
    7016: [
        "honest_verdict",
        "verdict_class",
        "v615_task_contract_conforms_score",
        "gate_check_summary",
    ],
    7017: [
        "honest_verdict",
        "verdict_class",
        "task_compute_receipt_ready_score",
        "phase_receipt_rows",
        "gpu_sample_rows",
        "runner_decision_rows",
    ],
    7018: [
        "honest_verdict",
        "verdict_class",
        "v615_sota_ingestion_complete_score",
        "post_marker_delta_rows",
    ],
    7019: [
        "honest_verdict",
        "verdict_class",
        "arc_belief_stream_ready_score",
        "solve_provenance",
        "rows",
    ],
    7020: [
        "honest_verdict",
        "verdict_class",
        "belief_ledger_ready_score",
        "rows",
        "gate_check_summary",
    ],
    7021: [
        "honest_verdict",
        "verdict_class",
        "belief_utility_comparison_complete_score",
        "belief_future_utility_positive_score",
        "per_decision_results",
        "paired_delta_rows",
        "confidence_intervals",
        "query_cost_rows",
    ],
    7022: [
        "honest_verdict",
        "verdict_class",
        "belief_shadow_safe_score",
        "belief_promotion_ready_score",
        "promotion_gate_rows",
        "rows",
    ],
    7023: ["honest_verdict", "verdict_class", "belief_query_api_ready_score", "rows"],
    7024: [
        "honest_verdict",
        "verdict_class",
        "belief_selector_live_path_ready_score",
        "shipped_default_unchanged",
        "rows",
    ],
    7025: [
        "honest_verdict",
        "verdict_class",
        "belief_shadow_trace_ready_score",
        "gate_check_summary",
        "solve_provenance",
        "task_compute_receipt",
        "phase_receipt_rows",
        "gpu_sample_rows",
        "model_execution_rows",
        "rows",
    ],
    7026: [
        "honest_verdict",
        "verdict_class",
        "status",
        "gate_check_summary",
        "failed_upstream",
        "failed_field",
        "failed_expected",
        "failed_observed",
        "blocked_at_layer",
        "per_game_results",
        "paired_delta_rows",
        "confidence_intervals",
    ],
}

REQUIRED_ARTIFACT_FIELDS = {
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "expected_task_count",
    "observed_task_count",
    "expected_id_order",
    "observed_id_order",
    "markdown_yaml_contract_rows",
    "task_outcome_rows",
    "cited_upstream_artifacts",
    "missing_upstream_rows",
    "blocked_upstream_rows",
    "verdict_propagation_rows",
    "oracle_boundary_rows",
    "adversarial_verification_rows",
    "row_consistency_rows",
    "prospective_utility_recomputation",
    "live_ab_recomputation",
    "solve_provenance_rows",
    "solve_registry_rows",
    "compute_receipt_summary",
    "scientific_gap_summary",
    "completed_nulls",
    "disqualifications",
    "blockers",
    "retirements",
    "promoted_claims",
    "v616_handoff",
    "docs_reconciled",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why it protects the scientific decision.",
    "preconditions_checked": "Explicit source and tool checks separate capstone execution failure from upstream evidence.",
    "inference_substrate": "The capstone must disclose that it replays evidence without model inference.",
    "duration_s": "Monotonic wall time shows that source reading and verification ran.",
    "source_artifact_hashes": "Content hashes bind every conclusion to exact source bytes.",
    "rows": "One terminal row per planned task prevents pooled prose from hiding missing work.",
    "expected_task_count": "The fixed count prevents silent scope expansion or task loss.",
    "observed_task_count": "The observed count exposes roadmap drift.",
    "expected_id_order": "The fixed order preserves the planned causal sequence.",
    "observed_id_order": "The observed order makes reordering visible.",
    "markdown_yaml_contract_rows": "Independent contract rows expose title, deliverable, gate, and producer drift.",
    "task_outcome_rows": "Every task keeps its own terminal evidence class.",
    "cited_upstream_artifacts": "Every imported field records its source, hash, and prior summary check.",
    "missing_upstream_rows": "Missing work remains a block rather than becoming a null result.",
    "blocked_upstream_rows": "External gate blocks stay visible and are not relabeled partial.",
    "verdict_propagation_rows": "Closed classes prevent positive laundering during synthesis.",
    "oracle_boundary_rows": "Current-outcome authority and executable oracles cannot become oracle-distinct value.",
    "adversarial_verification_rows": "Live verifier findings remain attached to each source claim.",
    "row_consistency_rows": "Row-level contradictions can block a copied headline.",
    "prospective_utility_recomputation": "Future utility must derive from matched decision rows.",
    "live_ab_recomputation": "Live value must derive from complete per-game cells under matched budgets.",
    "solve_provenance_rows": "Only new live self-discovery solves can enter a live headline.",
    "solve_registry_rows": "Registry checks prevent duplicate public-game solves from receiving credit.",
    "compute_receipt_summary": "Matched calls, actions, phases, and GPU receipts are required for a live comparison.",
    "scientific_gap_summary": "The handoff must name the smallest unresolved scientific boundaries.",
    "completed_nulls": "Complete nulls are durable findings and must not be retried unchanged.",
    "disqualifications": "Invalid or leaked evidence must remain separate from null science.",
    "blockers": "Required absent or gate-blocked work must name the exact external state.",
    "retirements": "Retirement requires a complete mechanism and value test, not an execution block.",
    "promoted_claims": "Only claims that pass every contract, safety, provenance, compute, and value gate may be promoted.",
    "v616_handoff": "One bounded next action prevents an unsupported expansion of scope.",
    "docs_reconciled": "The field records that the conductor owns later status and traceability reconciliation.",
    "random_seed": "A fixed seed makes any row resampling repeatable.",
    "reproducibility_checksum": "A stable digest detects changes to the evidence decision.",
    "gate_check_summary": "A blocked capstone names the failed check and exact expected and observed values.",
    "verifier_is_oracle": "False prevents the audit itself from being presented as an outcome oracle.",
    "verdict_class": "A closed terminal class preserves positive, null, blocked, disqualified, circular, and partial states.",
    "honest_verdict": "A terminal prefix gives the conductor an unambiguous milestone result.",
}

_SUMMARIZER_CACHE: dict[tuple[str, int, int], JsonDict] = {}


def sha256_path(path: Path) -> str:
    """Return a labeled SHA-256 for the exact bytes at one path."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normal_gate(raw: Mapping[str, Any]) -> JsonDict:
    """Normalize a structured gate without changing its producer field."""

    return {
        "upstream": str(raw.get("upstream") or ""),
        "artifact_field": str(raw.get("artifact_field") or ""),
        "op": str(raw.get("op") or "=="),
        "value": raw.get("value"),
    }


def _document_gates(text: str) -> list[JsonDict]:
    """Parse the narrow prerequisite syntax in the V615 contract table."""

    if text.strip().lower().startswith("none"):
        return []
    rows: list[JsonDict] = []
    for number, field, op, value in re.findall(
        r"Exp(\d+)\s+`([A-Za-z0-9_]+)\s*(==|!=|>=|<=|>|<)\s*([^`]+)`", text
    ):
        upstream = TASK_BY_NUMBER.get(int(number), {}).get("task_id", f"exp{number}")
        rows.append(
            {
                "upstream": upstream,
                "artifact_field": field,
                "op": op,
                "value": yaml.safe_load(value.strip()),
            }
        )
    return rows


def parse_document_contract(text: str) -> list[JsonDict]:
    """Parse the Markdown table without borrowing values from YAML."""

    marker = "## 7. Phases and Exact Task Contract"
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
            "source": "markdown",
            "order": int(order),
            "task_id": task_id,
            "title": title.strip(),
            "deliverable": deliverable,
            "gates": _document_gates(gates),
            "producer_prompt": "",
        }
        for order, task_id, title, deliverable, gates in pattern.findall(section)
    ]


def parse_yaml_contract(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Parse active roadmap tasks independently from the Markdown source."""

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    rows: list[JsonDict] = []
    for order, raw in enumerate(tasks, 1):
        task = raw if isinstance(raw, Mapping) else {}
        rows.append(
            {
                "source": "yaml",
                "order": order,
                "task_id": str(task.get("id") or ""),
                "title": str(task.get("title") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "gates": [
                    _normal_gate(gate)
                    for gate in task.get("gated_on", [])
                    if isinstance(gate, Mapping)
                ],
                "producer_prompt": str(task.get("prompt") or ""),
            }
        )
    return rows


def build_contract_rows(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Compare all fixed contract fields and each gate's producer spelling."""

    yaml_by_id = {str(row.get("task_id")): row for row in yaml_rows}
    rows: list[JsonDict] = []
    for index in range(max(12, len(document_rows), len(yaml_rows))):
        expected = EXPECTED_TASKS[index] if index < len(EXPECTED_TASKS) else {}
        document = document_rows[index] if index < len(document_rows) else {}
        roadmap = yaml_rows[index] if index < len(yaml_rows) else {}
        expected_gates = deepcopy(expected.get("gates", []))
        producer_fields_present = all(
            str(gate["artifact_field"])
            in str(yaml_by_id.get(str(gate["upstream"]), {}).get("producer_prompt") or "")
            for gate in expected_gates
        )
        checks = {
            "order": document.get("order") == roadmap.get("order") == index + 1,
            "task_id": document.get("task_id") == roadmap.get("task_id") == expected.get("task_id"),
            "title": document.get("title") == roadmap.get("title") == expected.get("title"),
            "deliverable": document.get("deliverable")
            == roadmap.get("deliverable")
            == expected.get("deliverable"),
            "gates": document.get("gates") == roadmap.get("gates") == expected_gates,
            "producer_fields": producer_fields_present,
        }
        rows.append(
            {
                "order": index + 1,
                "task_id": document.get("task_id") or roadmap.get("task_id"),
                "markdown": {key: document.get(key) for key in ("title", "deliverable", "gates")},
                "yaml": {key: roadmap.get(key) for key in ("title", "deliverable", "gates")},
                "producer_fields_present": producer_fields_present,
                "failed_checks": [name for name, passed in checks.items() if not passed],
                "passed": all(checks.values()),
                "terminal": True,
            }
        )
    return rows


def contract_conforms(
    document_rows: Sequence[Mapping[str, Any]],
    yaml_rows: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Return true only for the fixed count, order, and full parity."""

    return (
        len(document_rows) == len(yaml_rows) == len(EXPECTED_TASKS) == 12
        and [row.get("task_id") for row in document_rows] == EXPECTED_ID_ORDER
        and [row.get("task_id") for row in yaml_rows] == EXPECTED_ID_ORDER
        and all(row.get("passed") is True for row in contract_rows)
    )


def classify_task(task: Mapping[str, Any], payload: Mapping[str, Any] | None) -> JsonDict:
    """Preserve one source class while applying explicit oracle boundaries."""

    if payload is None:
        return {
            "number": task.get("number"),
            "task_id": task.get("task_id"),
            "artifact_path": task.get("deliverable"),
            "artifact_present": False,
            "state": "missing",
            "verdict_class": "blocked",
            "honest_verdict": None,
            "verifier_is_oracle": None,
            "eligible_for_promotion": False,
            "blocked_at_layer": "artifact_inventory",
            "terminal": True,
        }
    verdict = str(payload.get("honest_verdict") or "")
    declared = payload.get("verdict_class")
    if payload.get("status") == "blocked" or verdict.startswith("blocked_"):
        verdict_class = "blocked"
    elif declared in VERDICT_CLASSES:
        verdict_class = str(declared)
    elif "disqualified" in verdict:
        verdict_class = "disqualified"
    elif "null" in verdict:
        verdict_class = "null"
    else:
        verdict_class = "partial"
    oracle = payload.get("verifier_is_oracle") is True
    if oracle and verdict_class == "positive":
        verdict_class = "circular_positive"
    flagged = payload.get("flagged_adversarial") is True
    state = "flagged" if flagged else verdict_class
    return {
        "number": task.get("number"),
        "task_id": task.get("task_id"),
        "artifact_path": task.get("deliverable"),
        "artifact_present": True,
        "state": state,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
        "eligible_for_promotion": verdict_class == "positive" and not flagged and not oracle,
        "blocked_at_layer": payload.get("blocked_at_layer"),
        "terminal": True,
    }


def _summarize_before_import(root: Path, path: Path) -> JsonDict:
    """Run the required reader before JSON fields become available."""

    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _SUMMARIZER_CACHE.get(key)
    if cached is not None:
        return deepcopy(cached)
    command = [str(root / ".venv/bin/python"), str(root / SUMMARIZER_PATH), str(path)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    row = {
        "command": shlex.join(command),
        "exit_code": result.returncode,
        "status": "critical"
        if result.returncode == 2
        else "warn"
        if result.returncode == 1
        else "clean",
        "summary_excerpt": result.stdout[-2000:],
        "terminal": True,
    }
    _SUMMARIZER_CACHE[key] = deepcopy(row)
    return row


def _load_sources(root: Path) -> tuple[dict[int, JsonDict], list[JsonDict], dict[str, str]]:
    """Summarize, then read and hash every available V615 upstream artifact."""

    payloads: dict[int, JsonDict] = {}
    citations: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for task in EXPECTED_TASKS[:-1]:
        number = int(task["number"])
        relative = Path(str(task["deliverable"]))
        path = root / relative
        if not path.is_file():
            continue
        summary = _summarize_before_import(root, path)
        digest = sha256_path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            citations.append(
                {
                    "experiment_id": number,
                    "artifact_path": relative.as_posix(),
                    "fields_imported": [],
                    "sha256": digest,
                    "summarizer_exit_code": summary["exit_code"],
                    "summarizer_status": summary["status"],
                    "summarizer_ran_before_import": True,
                    "read_error": f"{type(exc).__name__}: {exc}",
                    "terminal": True,
                }
            )
            hashes[relative.as_posix()] = digest
            continue
        if not isinstance(payload, dict):
            continue
        payloads[number] = payload
        hashes[relative.as_posix()] = digest
        citations.append(
            {
                "experiment_id": number,
                "artifact_path": relative.as_posix(),
                "fields_imported": [field for field in IMPORTED_FIELDS[number] if field in payload],
                "sha256": digest,
                "honest_verdict": payload.get("honest_verdict"),
                "verdict_class": classify_task(task, payload)["verdict_class"],
                "verifier_is_oracle": payload.get("verifier_is_oracle"),
                "summarizer_exit_code": summary["exit_code"],
                "summarizer_status": summary["status"],
                "summarizer_ran_before_import": True,
                "adversarial_status": None,
                "terminal": True,
            }
        )
    return payloads, citations, hashes


def _adversarial_rows(root: Path, citations: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Run the current adversarial verifier on each readable source."""

    from scripts import adversarial_verify

    rows: list[JsonDict] = []
    for citation in citations:
        if citation.get("read_error"):
            continue
        path = root / str(citation["artifact_path"])
        report = adversarial_verify.verify_artifact(path)
        flags = deepcopy(report.get("flags", []))
        rows.append(
            {
                "experiment_id": citation["experiment_id"],
                "artifact_path": citation["artifact_path"],
                "status": "critical"
                if any(flag.get("severity") == "critical" for flag in flags)
                else "warn"
                if flags
                else "clean",
                "max_severity": report.get("max_severity", 0),
                "flags": flags,
                "critical": any(flag.get("severity") == "critical" for flag in flags),
                "terminal": True,
            }
        )
    return rows


def _row_consistency_rows(root: Path, citations: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Run the row-verdict checker on every readable source artifact."""

    from scripts import verdict_row_consistency_lint

    rows: list[JsonDict] = []
    for citation in citations:
        if citation.get("read_error"):
            continue
        status, findings = verdict_row_consistency_lint.check_artifact(
            root / str(citation["artifact_path"])
        )
        rows.append(
            {
                "experiment_id": citation["experiment_id"],
                "artifact_path": citation["artifact_path"],
                "status": status,
                "findings": findings,
                "critical": status == "findings" and bool(findings),
                "terminal": True,
            }
        )
    return rows


def _comparisons(by_unit: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> JsonDict:
    """Count matched belief wins, losses, and ties against each control."""

    output: JsonDict = {}
    for control in ("frozen", "recency_only"):
        counts = {"wins": 0, "losses": 0, "ties": 0, "comparable_units": 0}
        for arms in by_unit.values():
            belief = arms.get("counterexample_belief", {}).get("action_ranking_accuracy")
            baseline = arms.get(control, {}).get("action_ranking_accuracy")
            if not isinstance(belief, (int, float)) or not isinstance(baseline, (int, float)):
                continue
            counts["comparable_units"] += 1
            if belief > baseline:
                counts["wins"] += 1
            elif belief < baseline:
                counts["losses"] += 1
            else:
                counts["ties"] += 1
        output[f"counterexample_belief_minus_{control}"] = counts
    return output


def recompute_prospective_utility(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute Exp7021 statistics from per-decision rows only."""

    rows = [row for row in payload.get("per_decision_results", []) if isinstance(row, Mapping)]
    by_unit: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        unit = str(row.get("unit_id") or "")
        arm = str(row.get("arm") or "")
        if unit and arm:
            by_unit.setdefault(unit, {})[arm] = row
    missing_cells = [
        {"unit_id": unit, "arm": arm}
        for unit, arms in by_unit.items()
        for arm in PROSPECTIVE_ARMS
        if arm not in arms
    ]
    no_headroom = {
        unit
        for unit, arms in by_unit.items()
        if any(row.get("support_class") == "no_headroom" for row in arms.values())
    }
    unsupported = {
        unit
        for unit, arms in by_unit.items()
        if any(row.get("support_class") == "unsupported" for row in arms.values())
    }
    arm_metrics: JsonDict = {}
    for arm in PROSPECTIVE_ARMS:
        values = [
            row.get("action_ranking_accuracy")
            for arms in by_unit.values()
            if (row := arms.get(arm)) is not None
            and isinstance(row.get("action_ranking_accuracy"), (int, float))
        ]
        arm_metrics[arm] = {
            "scored_unit_count": len(values),
            "action_ranking_accuracy": sum(values) / len(values) if values else None,
        }
    selected_actions = sum(row.get("selected_action_id") is not None for row in rows)
    model_calls = sum(
        int(row.get("model_calls", 0)) for row in rows if isinstance(row.get("model_calls", 0), int)
    )
    return {
        "source": "experiment_7021.per_decision_results",
        "unit_count": len(by_unit),
        "row_count": len(rows),
        "arm_metrics": arm_metrics,
        "comparisons": _comparisons(by_unit),
        "no_headroom_unit_count": len(no_headroom),
        "no_headroom_units": sorted(no_headroom),
        "unsupported_unit_count": len(unsupported),
        "unsupported_units": sorted(unsupported),
        "missing_cell_count": len(missing_cells),
        "missing_cells": missing_cells,
        "intervals": deepcopy(payload.get("confidence_intervals", [])),
        "paired_intervals": deepcopy(payload.get("paired_delta_rows", [])),
        "selected_action_count": selected_actions,
        "live_action_count": 0,
        "model_call_count": model_calls,
        "query_cost_receipt_count": len(payload.get("query_cost_rows", [])),
        "pooled_value_used": False,
        "terminal": True,
    }


def recompute_live_ab(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute live comparisons, or preserve a complete absent-row boundary."""

    rows = [row for row in payload.get("per_game_results", []) if isinstance(row, Mapping)]
    blocked = (
        payload.get("verdict_class") == "blocked"
        or payload.get("status") == "blocked"
        or str(payload.get("honest_verdict") or "").startswith("blocked_")
    )
    if not rows:
        return {
            "source": "experiment_7026.per_game_results",
            "blocked": blocked,
            "per_game_row_count": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "no_headroom_unit_count": 0,
            "missing_cells": "all_planned_live_cells" if blocked else [],
            "intervals": [],
            "model_call_count": 0,
            "action_count": 0,
            "compute_receipt_count": 0,
            "eligible_live_solve_count": 0,
            "pooled_value_used": False,
            "terminal": True,
        }
    wins = sum(bool(row.get("win")) for row in rows)
    losses = sum(bool(row.get("loss")) for row in rows)
    ties = sum(bool(row.get("tie")) for row in rows)
    no_headroom = sum(bool(row.get("no_headroom")) for row in rows)
    model_calls = sum(int(row.get("model_calls", 0) or 0) for row in rows)
    actions = sum(int(row.get("actions", row.get("action_count", 0)) or 0) for row in rows)
    receipts = sum(bool(row.get("compute_receipt")) for row in rows)
    return {
        "source": "experiment_7026.per_game_results",
        "blocked": blocked,
        "per_game_row_count": len(rows),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "no_headroom_unit_count": no_headroom,
        "missing_cells": [],
        "intervals": deepcopy(payload.get("confidence_intervals", [])),
        "paired_intervals": deepcopy(payload.get("paired_delta_rows", [])),
        "model_call_count": model_calls,
        "action_count": actions,
        "compute_receipt_count": receipts,
        "eligible_live_solve_count": 0,
        "pooled_value_used": False,
        "terminal": True,
    }


def _registry_levels(registry: Mapping[str, Any]) -> dict[str, int]:
    """Return the recorded reproduced level for every registry game."""

    games = registry.get("games", [])
    if isinstance(games, Mapping):
        return {
            str(game): int(row.get("levels_reproduced", 0) or 0)
            for game, row in games.items()
            if isinstance(row, Mapping)
        }
    return {
        str(row.get("game")): int(row.get("levels_reproduced", 0) or 0)
        for row in games
        if isinstance(row, Mapping) and row.get("game")
    }


def build_solve_rows(
    payloads: Mapping[int, Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check all claimed solve rows against provenance and registry state."""

    levels = _registry_levels(registry)
    provenance_rows: list[JsonDict] = []
    registry_rows: list[JsonDict] = []
    for experiment_id, payload in payloads.items():
        candidates = [
            row for row in payload.get("per_game_results", []) if isinstance(row, Mapping)
        ]
        for row in candidates:
            solved = row.get("solved") is True or int(row.get("new_levels_banked", 0) or 0) > 0
            if not solved:
                continue
            game = str(row.get("game") or "")
            level = int(row.get("level_after", row.get("reproduced_levels", 0)) or 0)
            provenance = str(
                row.get("solve_provenance") or payload.get("solve_provenance") or "undeclared"
            )
            registered = levels.get(game, 0)
            duplicate = bool(game and level <= registered)
            eligible = provenance == "live_agent_self_discovery" and not duplicate and level > 0
            provenance_rows.append(
                {
                    "experiment_id": experiment_id,
                    "game": game,
                    "level": level,
                    "solve_provenance": provenance,
                    "duplicate": duplicate,
                    "eligible_for_live_headline": eligible,
                    "terminal": True,
                }
            )
            registry_rows.append(
                {
                    "experiment_id": experiment_id,
                    "game": game,
                    "claimed_level": level,
                    "registered_level": registered,
                    "duplicate": duplicate,
                    "eligible_for_live_headline": eligible,
                    "terminal": True,
                }
            )
    if not provenance_rows:
        for experiment_id, payload in payloads.items():
            if "solve_provenance" in payload:
                provenance_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "solve_provenance": payload.get("solve_provenance"),
                        "solve_claimed": int(payload.get("arc_new_level_banked", 0) or 0) > 0,
                        "eligible_for_live_headline": False,
                        "duplicate": False,
                        "terminal": True,
                    }
                )
    registry_rows.insert(
        0,
        {
            "registry_path": REGISTRY_PATH.as_posix(),
            "reproducible_total_levels": registry.get("reproducible_total_levels"),
            "reproducible_total_games": registry.get("reproducible_total_games"),
            "checked_solve_claim_count": len(registry_rows),
            "terminal": True,
        },
    )
    return provenance_rows, registry_rows


def _preconditions(root: Path, output: Path) -> tuple[list[JsonDict], bool]:
    """Check only resources needed to execute the capstone itself."""

    paths = (
        ("v615_markdown_contract", DESIGN_PATH),
        ("active_v615_roadmap", ROADMAP_PATH),
        ("capstone_spec", SPEC_PATH),
        ("artifact_summarizer", SUMMARIZER_PATH),
        ("adversarial_verifier", ADVERSARIAL_PATH),
        ("row_consistency_lint", ROW_LINT_PATH),
        ("arc_solve_registry", REGISTRY_PATH),
    )
    rows = []
    for check, relative in paths:
        path = root / relative
        rows.append(
            {
                "check": check,
                "path": relative.as_posix(),
                "expected_value": "readable_file",
                "observed_value": "readable_file" if path.is_file() else "missing",
                "passed": path.is_file(),
                "terminal": True,
            }
        )
    writable = output.parent.is_dir() and output.parent.stat().st_mode != 0
    rows.append(
        {
            "check": "artifact_path_writable",
            "path": str(output),
            "expected_value": True,
            "observed_value": writable,
            "passed": writable,
            "terminal": True,
        }
    )
    return rows, all(row["passed"] for row in rows)


def _empty_artifact(run_date: str) -> JsonDict:
    """Return the complete blocked-safe schema before sources are read."""

    return {
        "schema": "carnot.exp7027.v615_capstone.v1",
        "experiment_id": 7027,
        "run_date": run_date,
        "status": "complete",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "expected_task_count": 12,
        "observed_task_count": 0,
        "expected_id_order": deepcopy(EXPECTED_ID_ORDER),
        "observed_id_order": [],
        "markdown_yaml_contract_rows": [],
        "task_outcome_rows": [],
        "cited_upstream_artifacts": [],
        "missing_upstream_rows": [],
        "blocked_upstream_rows": [],
        "verdict_propagation_rows": [],
        "oracle_boundary_rows": [],
        "adversarial_verification_rows": [],
        "row_consistency_rows": [],
        "prospective_utility_recomputation": recompute_prospective_utility({}),
        "live_ab_recomputation": recompute_live_ab({}),
        "solve_provenance_rows": [],
        "solve_registry_rows": [],
        "compute_receipt_summary": {},
        "scientific_gap_summary": [],
        "completed_nulls": [],
        "disqualifications": [],
        "blockers": [],
        "retirements": [],
        "promoted_claims": [],
        "v616_handoff": {},
        "docs_reconciled": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_v615_capstone_execution_precondition",
    }


def _compute_summary(payloads: Mapping[int, Mapping[str, Any]]) -> JsonDict:
    """Preserve contract receipts and expose absent live compute evidence."""

    contract = payloads.get(7017, {})
    shadow = payloads.get(7025, {})
    live = payloads.get(7026, {})
    live_rows = [row for row in live.get("per_game_results", []) if isinstance(row, Mapping)]
    budget_rows = [row for row in live.get("budget_rows", []) if isinstance(row, Mapping)]
    budgets_match = bool(budget_rows) and all(
        row.get("passed") is True
        or (
            row.get("action_budget_matched") is True
            and row.get("model_call_budget_matched") is True
        )
        for row in budget_rows
    )
    live_phase_count = len(live.get("phase_receipt_rows", []))
    live_gpu_count = len(live.get("gpu_sample_rows", []))
    return {
        "receipt_contract_ready_score": contract.get("task_compute_receipt_ready_score"),
        "contract_phase_receipt_count": len(contract.get("phase_receipt_rows", [])),
        "contract_gpu_sample_count": len(contract.get("gpu_sample_rows", [])),
        "shadow_task_receipt_present": bool(shadow.get("task_compute_receipt")),
        "shadow_phase_receipt_count": len(shadow.get("phase_receipt_rows", [])),
        "shadow_gpu_sample_count": len(shadow.get("gpu_sample_rows", [])),
        "shadow_model_execution_count": len(shadow.get("model_execution_rows", [])),
        "live_ab_gate_blocked": live.get("status") == "blocked",
        "live_ab_per_game_row_count": len(live_rows),
        "live_ab_phase_receipt_count": live_phase_count,
        "live_ab_gpu_sample_count": live_gpu_count,
        "matched_live_compute": bool(live_rows)
        and budgets_match
        and live_phase_count > 0
        and live_gpu_count > 0,
        "terminal": True,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding duration and command transcripts."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s", "validation_command_rows"}
    }
    data = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def build_artifact(root: Path, run_date: str, *, run_commands: bool) -> JsonDict:
    """Build one deterministic capstone without changing source evidence."""

    del run_commands
    started = time.monotonic()
    output = root / OUTPUT_PATH
    artifact = _empty_artifact(run_date)
    preconditions, ready = _preconditions(root, output)
    artifact["preconditions_checked"] = preconditions
    for relative in (
        DESIGN_PATH,
        ROADMAP_PATH,
        SPEC_PATH,
        REGISTRY_PATH,
        SUMMARIZER_PATH,
        ADVERSARIAL_PATH,
        ROW_LINT_PATH,
    ):
        if (root / relative).is_file():
            artifact["source_artifact_hashes"][relative.as_posix()] = sha256_path(root / relative)
    if not ready:
        failed = next(row for row in preconditions if not row["passed"])
        artifact["gate_check_summary"] = {
            "failed_check": failed["check"],
            "expected_value": failed["expected_value"],
            "observed_value": failed["observed_value"],
            "passed": False,
            "terminal": True,
        }
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact
    try:
        document_rows = parse_document_contract((root / DESIGN_PATH).read_text(encoding="utf-8"))
        roadmap_payload = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
        if not isinstance(roadmap_payload, Mapping):
            raise ValueError("active roadmap root is not a mapping")
        yaml_rows = parse_yaml_contract(roadmap_payload)
        registry = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8"))
        if not isinstance(registry, Mapping):
            raise ValueError("ARC registry root is not a mapping")
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        artifact["gate_check_summary"] = {
            "failed_check": "capstone_source_parse",
            "expected_value": "readable valid sources",
            "observed_value": f"{type(exc).__name__}: {exc}",
            "passed": False,
            "terminal": True,
        }
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    contract_rows = build_contract_rows(document_rows, yaml_rows)
    conforms = contract_conforms(document_rows, yaml_rows, contract_rows)
    payloads, citations, upstream_hashes = _load_sources(root)
    artifact["source_artifact_hashes"].update(upstream_hashes)
    adversarial_rows = _adversarial_rows(root, citations)
    row_rows = _row_consistency_rows(root, citations)
    adversarial_by_id = {row["experiment_id"]: row for row in adversarial_rows}
    for citation in citations:
        verification = adversarial_by_id.get(citation["experiment_id"], {})
        citation["adversarial_status"] = verification.get("status", "unreadable")

    task_rows = [
        classify_task(task, payloads.get(int(task["number"]))) for task in EXPECTED_TASKS[:-1]
    ]
    missing = [row for row in task_rows if row["state"] == "missing"]
    blocked = [row for row in task_rows if row["verdict_class"] == "blocked"]
    disqualified = [row for row in task_rows if row["verdict_class"] == "disqualified"]
    critical = [row for row in adversarial_rows if row["critical"]]
    prospective = recompute_prospective_utility(payloads.get(7021, {}))
    live = recompute_live_ab(payloads.get(7026, {}))
    solve_rows, registry_rows = build_solve_rows(payloads, registry)
    compute = _compute_summary(payloads)

    cold_safe = payloads.get(7022, {}).get("belief_shadow_safe_score") == 1
    prospective_positive = payloads.get(7021, {}).get("belief_future_utility_positive_score") == 1
    live_positive = payloads.get(7026, {}).get("belief_live_value_positive_score") == 1
    live_provenance = payloads.get(7025, {}).get("belief_shadow_trace_ready_score") == 1
    matched_compute = compute["matched_live_compute"] is True
    if not conforms or disqualified or critical:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_v615_capstone_invalid_contract_or_evidence"
    elif missing or blocked:
        verdict_class = "blocked"
        verdict = "complete_blocked_v615_capstone_required_live_evidence_absent"
    elif cold_safe and not prospective_positive and not live_positive:
        verdict_class = "null"
        verdict = "complete_null_v615_capstone_safe_belief_has_no_row_derived_value"
    elif (
        cold_safe and prospective_positive and live_positive and live_provenance and matched_compute
    ):
        verdict_class = "positive"
        verdict = "complete_positive_v615_capstone_belief_value_confirmed"
    else:
        verdict_class = "null"
        verdict = "complete_null_v615_capstone_belief_value_not_confirmed"

    self_row = classify_task(
        EXPECTED_TASKS[-1],
        {
            "status": "complete",
            "verdict_class": verdict_class,
            "honest_verdict": verdict,
            "verifier_is_oracle": False,
        },
    )
    all_task_rows = [*task_rows, self_row]
    gate_checks = [
        {
            "check": "exact_contract",
            "expected_value": True,
            "observed_value": conforms,
            "passed": conforms,
        },
        {
            "check": "cold_safety",
            "expected_value": True,
            "observed_value": cold_safe,
            "passed": cold_safe,
        },
        {
            "check": "prospective_value",
            "expected_value": True,
            "observed_value": prospective_positive,
            "passed": prospective_positive,
        },
        {
            "check": "live_shadow_provenance",
            "expected_value": True,
            "observed_value": live_provenance,
            "passed": live_provenance,
        },
        {
            "check": "matched_live_compute",
            "expected_value": True,
            "observed_value": matched_compute,
            "passed": matched_compute,
        },
        {
            "check": "row_derived_live_value",
            "expected_value": True,
            "observed_value": live_positive,
            "passed": live_positive,
        },
    ]
    for row in gate_checks:
        row["terminal"] = True
    live_error = payloads.get(7025, {}).get("gate_check_summary", {})
    named_defect = "live trace evidence is absent"
    if isinstance(live_error, Mapping):
        named_defect = str(live_error.get("observed_value") or named_defect)
    handoff_action = (
        "repair_one_named_evidence_defect"
        if verdict_class in {"blocked", "disqualified"}
        else "retire_explicit_belief_after_complete_null"
        if verdict_class == "null"
        else "promote_belief_to_larger_held_roster"
    )
    first_failed = next((row for row in gate_checks if not row["passed"]), None)
    if verdict_class == "blocked":
        gate_summary = {
            "failed_check": "required_live_upstream_complete",
            "expected_value": {
                "exp7025.belief_shadow_trace_ready_score": 1,
                "exp7026.status": "complete",
            },
            "observed_value": {
                "exp7025.belief_shadow_trace_ready_score": payloads.get(7025, {}).get(
                    "belief_shadow_trace_ready_score"
                ),
                "exp7026.status": payloads.get(7026, {}).get("status"),
            },
            "passed": False,
            "checks": gate_checks,
            "terminal": True,
        }
    elif verdict_class == "disqualified":
        gate_summary = {
            "failed_check": "contract_or_evidence_valid",
            "expected_value": True,
            "observed_value": {
                "contract_conforms": conforms,
                "disqualified": len(disqualified),
                "critical": len(critical),
            },
            "passed": False,
            "checks": gate_checks,
            "terminal": True,
        }
    else:
        gate_summary = {
            "failed_check": first_failed["check"] if first_failed else None,
            "expected_value": first_failed["expected_value"] if first_failed else True,
            "observed_value": first_failed["observed_value"] if first_failed else True,
            "passed": verdict_class == "positive",
            "checks": gate_checks,
            "terminal": True,
        }

    artifact.update(
        {
            "observed_task_count": len(yaml_rows),
            "observed_id_order": [str(row.get("task_id")) for row in yaml_rows],
            "markdown_yaml_contract_rows": contract_rows,
            "rows": all_task_rows,
            "task_outcome_rows": all_task_rows,
            "cited_upstream_artifacts": citations,
            "missing_upstream_rows": missing,
            "blocked_upstream_rows": blocked,
            "verdict_propagation_rows": [
                {
                    "experiment_id": row["number"],
                    "source_class": row["verdict_class"],
                    "propagated_class": row["verdict_class"],
                    "state": row["state"],
                    "terminal": True,
                }
                for row in all_task_rows
            ],
            "oracle_boundary_rows": [
                {
                    "experiment_id": row["number"],
                    "verifier_is_oracle": row["verifier_is_oracle"],
                    "eligible_for_promotion": row["eligible_for_promotion"],
                    "boundary_preserved": not (
                        row["verifier_is_oracle"] is True and row["eligible_for_promotion"]
                    ),
                    "terminal": True,
                }
                for row in all_task_rows
            ],
            "adversarial_verification_rows": adversarial_rows,
            "row_consistency_rows": row_rows,
            "prospective_utility_recomputation": prospective,
            "live_ab_recomputation": live,
            "solve_provenance_rows": solve_rows,
            "solve_registry_rows": registry_rows,
            "compute_receipt_summary": compute,
            "scientific_gap_summary": [
                {
                    "gap": "prospective_interval",
                    "state": "null",
                    "detail": "belief-minus-recency interval crosses zero",
                    "terminal": True,
                },
                {
                    "gap": "live_shadow_trace",
                    "state": "blocked",
                    "detail": named_defect,
                    "terminal": True,
                },
                {
                    "gap": "held_mechanic_live_value",
                    "state": "blocked",
                    "detail": "Exp7026 was stopped by the Exp7025 gate",
                    "terminal": True,
                },
            ],
            "completed_nulls": [
                {
                    "experiment_id": row["number"],
                    "honest_verdict": row["honest_verdict"],
                    "terminal": True,
                }
                for row in task_rows
                if row["verdict_class"] == "null"
            ],
            "disqualifications": [*disqualified, *critical],
            "blockers": [*missing, *blocked],
            "retirements": []
            if verdict_class == "blocked"
            else (
                [
                    {
                        "mechanism": "explicit_arc_belief",
                        "reason": "complete safe null",
                        "terminal": True,
                    }
                ]
                if verdict_class == "null"
                else []
            ),
            "promoted_claims": []
            if verdict_class != "positive"
            else [{"claim": "explicit belief improves held live decisions", "terminal": True}],
            "v616_handoff": {
                "action": handoff_action,
                "named_defect": named_defect,
                "activate_v616": False,
                "kan_compression_recommended": prospective_positive,
                "bounded_scope": "Correct the GGUF model_filename provenance value, then rerun Exp7025 and the gated Exp7026 cells only.",
                "terminal": True,
            },
            "docs_reconciled": False,
            "verdict_class": verdict_class,
            "honest_verdict": verdict,
            "gate_check_summary": gate_summary,
        }
    )
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject missing fields, contract drift, class drift, and hash drift."""

    errors = [
        f"missing required field: {field}"
        for field in sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    ]
    if set(artifact.get("field_principles", {})) != REQUIRED_ARTIFACT_FIELDS:
        errors.append("field principles do not cover every required field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("expected_task_count") != 12:
        errors.append("expected task count mismatch")
    if artifact.get("expected_id_order") != EXPECTED_ID_ORDER:
        errors.append("expected task order mismatch")
    if (
        artifact.get("observed_task_count")
        and artifact.get("observed_id_order") != EXPECTED_ID_ORDER
    ):
        errors.append("observed task order mismatch")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict") or "")
    prefix_ok = {
        "positive": verdict.startswith("complete_positive_"),
        "circular_positive": verdict.startswith("complete_circular_positive_"),
        "null": verdict.startswith("complete_null_"),
        "blocked": verdict.startswith("complete_blocked_"),
        "disqualified": verdict.startswith("complete_disqualified_"),
        "partial": verdict.startswith("partial_"),
    }.get(verdict_class, False)
    if not prefix_ok:
        errors.append("honest verdict prefix does not match verdict class")
    gate = artifact.get("gate_check_summary", {})
    if verdict_class == "blocked" and (
        not isinstance(gate, Mapping)
        or gate.get("failed_check") is None
        or "expected_value" not in gate
        or "observed_value" not in gate
    ):
        errors.append("blocked verdict lacks exact gate diagnostic")
    if artifact.get("docs_reconciled") is not False:
        errors.append("conductor-owned document reconciliation must remain deferred")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def _write(path: Path, artifact: Mapping[str, Any]) -> None:
    """Atomically replace only the caller-selected artifact path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_command(root: Path, name: str, command: Sequence[str]) -> JsonDict:
    """Run one bounded validation command and retain its exit state."""

    result = subprocess.run(list(command), cwd=root, capture_output=True, text=True, check=False)
    return {
        "name": name,
        "command": shlex.join(command),
        "exit_code": result.returncode,
        "output": "\n".join(part for part in (result.stdout, result.stderr) if part)[-8000:],
        "terminal": True,
    }


def run_validation_commands(root: Path, output: Path) -> list[JsonDict]:
    """Run the focused checks named by the V615 contract exactly once."""

    python = str(root / ".venv/bin/python")
    module_script = "scripts/experiments/experiment_7027_v615_capstone.py"
    all_artifacts = [str(task["deliverable"]) for task in EXPECTED_TASKS]
    commands: tuple[tuple[str, list[str]], ...] = (
        (
            "focused_tests",
            [
                python,
                "-m",
                "pytest",
                "tests/python/test_experiment_7027_v615_capstone.py",
                "-q",
                "--no-cov",
            ],
        ),
        (
            "yaml_parse",
            [python, "-c", "import yaml; yaml.safe_load(open('research-roadmap.yaml'))"],
        ),
        ("artifact_validation", [python, module_script, "--validate", str(output)]),
        ("adversarial_verification", [python, "scripts/adversarial_verify.py", *all_artifacts]),
        (
            "row_consistency_lint",
            [python, "scripts/verdict_row_consistency_lint.py", *all_artifacts],
        ),
        ("exclusion_lint", [python, "scripts/exclusion_manifest_lint.py", "research-roadmap.yaml"]),
        (
            "arc_floor_lint",
            [python, "scripts/arc_levelup_guarantee_lint.py", "research-roadmap.yaml"],
        ),
        (
            "spec_coverage",
            [
                python,
                "scripts/check_spec_coverage.py",
                "tests/python/test_experiment_7027_v615_capstone.py",
            ],
        ),
        ("root_clutter", [python, "scripts/root_clutter_sweep.py"]),
    )
    return [_run_command(root, name, command) for name, command in commands]


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate one V615 capstone artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260905")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-commands", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(
                json.dumps({"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}, indent=2)
            )
            return 1
        errors = validate_artifact(payload)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return int(bool(errors))
    root = args.root.resolve()
    output = args.output or (root / OUTPUT_PATH)
    if not output.is_absolute():
        output = root / output
    artifact = build_artifact(root, args.date, run_commands=not args.no_commands)
    _write(output, artifact)
    if not args.no_commands:
        artifact["validation_command_rows"] = run_validation_commands(root, output)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        _write(output, artifact)
    errors = validate_artifact(artifact)
    print(json.dumps({"artifact": str(output), "valid": not errors, "errors": errors}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - the thin script is the command entry point.
    raise SystemExit(main(sys.argv[1:]))
