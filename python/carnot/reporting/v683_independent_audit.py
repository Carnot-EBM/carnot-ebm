"""Audit V683 producer bytes without dispatching upstream experiments.

REQ-REPORT-7877 keeps evidence qualification separate from measured benefit.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import (
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)


AUTHORITY = "docs/research-notes/v683-authority-snapshots/staged.yaml"
SCIENCE = {7869, 7870, 7871, 7872, 7873, 7875}
READY = {
    7865: "contract_ready_score",
    7866: "source_boundary_ready_score",
    7867: "natural_training_ready_score",
    7868: "intervention_protocol_ready_score",
    7869: "energy_fit_ready_score",
    7870: "decision_evidence_ready_score",
    7871: "qwen_evidence_ready_score",
    7872: "learning_evidence_ready_score",
    7873: "feedback_evidence_ready_score",
    7874: "arc_delta_ready_score",
    7875: "service_evidence_ready_score",
    7876: "hardware_evidence_ready_score",
}
QUALIFIED = ("positive", "circular_positive", "null")


def _data(path: Path) -> dict[str, Any]:
    """Return only an object-shaped JSON artifact; absence remains observable."""
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _operand(
    number: int, path: Path, field: str, expected: Any, observed: Any, op: str = "=="
) -> dict[str, Any]:
    """Retain exact failed bytes and the operand that did not pass."""
    return {
        "upstream_id": f"Exp{number}",
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }


def _receipts(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Read both historical receipt shapes without changing check classes."""
    value = data.get("validation_receipts", [])
    if isinstance(value, dict):
        value = value.get("checks", [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def inspect_sources(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare exact task deliverables and declared gates in execution order."""
    authority = yaml.safe_load((root / AUTHORITY).read_text())
    tasks = authority["tasks"][:12]
    if [task["id"].split("-", 1)[0] for task in tasks] != [f"exp{n}" for n in range(7865, 7877)]:
        raise ValueError("v683_authority_order_changed")
    lookup = {task["id"]: task for task in tasks}
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for number, task in zip(range(7865, 7877), tasks, strict=True):
        path = root / task["deliverable"]
        data = _data(path)
        aliases = [
            p for p in sorted((root / "results").glob(f"experiment_{number}_*.json")) if p != path
        ]
        state = (
            "missing_producer"
            if not path.is_file()
            else "invalid_json"
            if not data
            else str(data.get("verdict_class", "missing_verdict_class"))
        )
        if not data:
            observed = "conductor_skip_receipt" if aliases and not path.is_file() else state
            failures.append(
                _operand(number, path, "science_producer", "current qualified artifact", observed)
            )
        else:
            for field, expected in (
                ("experiment_id", number),
                ("task_id", task["id"]),
                ("milestone", "2026.09.683"),
                ("run_date", "20260929"),
                ("flagged_adversarial", False),
                (READY[number], 1),
            ):
                if data.get(field) != expected:
                    failures.append(_operand(number, path, field, expected, data.get(field)))
            if data.get("verdict_class") not in QUALIFIED:
                failures.append(
                    _operand(
                        number,
                        path,
                        "verdict_class",
                        list(QUALIFIED),
                        data.get("verdict_class"),
                        "in",
                    )
                )
            for receipt in _receipts(data):
                kind = receipt.get("classification", receipt.get("class", receipt.get("scope")))
                if kind == "required" and (
                    receipt.get("exit_code") != 0
                    or receipt.get("passed") is False
                    or receipt.get("timed_out")
                ):
                    failures.append(
                        _operand(
                            number,
                            path,
                            f"validation_receipts.{receipt.get('name', 'unnamed')}",
                            0,
                            receipt.get("exit_code"),
                        )
                    )
        gate_failed = False
        for gate in task.get("gated_on", []):
            upstream = lookup[gate["upstream"]]
            upstream_path = root / upstream["deliverable"]
            upstream_data = _data(upstream_path)
            field = gate["artifact_field"]
            actual = upstream_data.get(
                field, "missing_field" if upstream_data else "missing_source"
            )
            expected = gate["value"]
            passed = actual == expected if gate["op"] == "==" else actual in expected
            if not passed:
                gate_failed = True
                failures.append(
                    _operand(
                        int(gate["upstream"][3:7]),
                        upstream_path,
                        field,
                        expected,
                        actual,
                        gate["op"],
                    )
                )
        own_failures = [
            f for f in failures if f["upstream_id"] == f"Exp{number}" and f["path"] == str(path)
        ]
        eligible = bool(data) and not own_failures and not gate_failed
        if eligible:
            state = "eligible_null" if data.get("verdict_class") == "null" else "eligible"
        rows.append(
            {
                "upstream_id": f"Exp{number}",
                "task_id": task["id"],
                "path": str(path),
                "hash": sha256_file(path) if path.is_file() else None,
                "status": state,
                "eligible": eligible,
                "role": "science_producer" if number in SCIENCE else "prerequisite_or_continuity",
                "source_exposure": "exposed_development"
                if number in SCIENCE
                else "administrative_or_historical",
                "label_authority": "original_human_label_only" if number in SCIENCE else "none",
                "skip_receipts": [
                    {"path": str(alias), "hash": sha256_file(alias), "role": "explanation_only"}
                    for alias in aliases
                ],
            }
        )
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path) if path.is_file() else None,
                "date": data.get("run_date"),
                "role": rows[-1]["role"],
                "exposure_status": rows[-1]["source_exposure"],
            }
        )
        sources.extend(
            {
                "path": str(alias),
                "sha256": sha256_file(alias),
                "date": None,
                "role": "conductor_skip_receipt",
                "exposure_status": "administrative",
            }
            for alias in aliases
        )
    return rows, failures, sources


def reduce_family_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average seeds within each family before reporting probability and cost."""
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed" and row.get("family_id") is not None:
            families[str(row["family_id"])].append(row)

    def mean_for(key: str) -> float | None:
        means = [
            sum(float(r[key]) for r in family if isinstance(r.get(key), (float, int)))
            / sum(isinstance(r.get(key), (float, int)) for r in family)
            for family in families.values()
            if any(isinstance(r.get(key), (float, int)) for r in family)
        ]
        return sum(means) / len(means) if means else None

    enriched = []
    for row in rows:
        risk = row.get("probability", row.get("probability_unsupported"))
        label = row.get("label")
        enriched.append(
            {
                **row,
                "brier": (risk - label) ** 2
                if isinstance(risk, (int, float)) and label in (0, 1)
                else None,
            }
        )
    families.clear()
    for row in enriched:
        if row.get("status") == "completed" and row.get("family_id") is not None:
            families[str(row["family_id"])].append(row)
    return {
        "independent_family_count": len(families),
        "seed_count": sum(len(group) for group in families.values()),
        "brier": mean_for("brier"),
        "cost": mean_for("cost"),
    }


def _primitive_rows(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project each observed unit and recompute only metrics present in raw rows."""
    units: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    keys = (
        "probability",
        "probability_unsupported",
        "label",
        "cost",
        "brier",
        "syntax_valid",
        "source_byte_fidelity",
        "semantic_sensitivity",
        "prediction_step",
        "feedback_step",
        "release_step",
        "admitted",
        "constraint_effect",
        "retention",
        "latency_ms",
        "metric",
        "current_hardware_execution",
        "new_level_solve",
    )
    for task in tasks:
        data = _data(Path(task["path"]))
        raw = data.get("rows", [])
        raw = raw if isinstance(raw, list) else []
        projected: list[dict[str, Any]] = []
        if not raw:
            projected.append(
                {
                    "upstream_id": task["upstream_id"],
                    "arm": task["task_id"],
                    "family_id": None,
                    "seed": None,
                    "status": task["status"],
                    "metric": None,
                    "eligible": 0,
                    "started": 0,
                    "completed": 0,
                    "censored": 0,
                    "excluded": 1,
                    "independent": 0,
                }
            )
        for index, row in enumerate(raw):
            if not isinstance(row, dict):
                row = {"status": "malformed", "metric": None}
            status = str(row.get("status", "unknown"))
            family = row.get("source_family", row.get("family_id", row.get("family")))
            projected.append(
                {
                    "upstream_id": task["upstream_id"],
                    "primitive_index": index,
                    "arm": row.get("arm"),
                    "family_id": family,
                    "seed": row.get("seed"),
                    "status": status,
                    "eligible": int(task["eligible"] and status != "excluded"),
                    "started": int(status not in ("unstarted", "excluded")),
                    "completed": int(status == "completed"),
                    "censored": int(status == "censored"),
                    "excluded": int(status == "excluded"),
                    "independent": 0,
                    "metric": {key: row[key] for key in keys if key in row},
                }
            )
        seen: set[str] = set()
        for row in projected:
            if row["family_id"] is not None:
                family_key = str(row["family_id"])
                row["independent"] = int(family_key not in seen and bool(row["eligible"]))
                seen.add(family_key)
        units.extend(projected)
        grouped = [
            {
                **r["metric"],
                "family_id": r["family_id"],
                "seed": r["seed"],
                "arm": r["arm"],
                "status": r["status"],
            }
            for r in projected
            if isinstance(r["metric"], dict)
        ]
        summary = reduce_family_rows(grouped)
        comparisons.append(
            {
                "upstream_id": task["upstream_id"],
                "path": task["path"],
                "hash": task["hash"],
                "source_exposure": task["source_exposure"],
                "label_authority": task["label_authority"],
                "row_count": len(raw),
                **summary,
                "qwen_parse_count": sum(
                    r["metric"].get("syntax_valid") is True
                    for r in projected
                    if isinstance(r["metric"], dict)
                ),
                "source_fidelity_count": sum(
                    r["metric"].get("source_byte_fidelity") is True
                    for r in projected
                    if isinstance(r["metric"], dict)
                ),
                "semantic_sensitivity_count": sum(
                    r["metric"].get("semantic_sensitivity") is True
                    for r in projected
                    if isinstance(r["metric"], dict)
                ),
                "temporal_release_violation_count": sum(
                    r["metric"].get("feedback_step", r["metric"].get("release_step", 10**12))
                    <= r["metric"].get("prediction_step", -1)
                    for r in projected
                    if isinstance(r["metric"], dict)
                ),
                "new_constraint_effect_count": sum(
                    bool(r["metric"].get("constraint_effect"))
                    for r in projected
                    if isinstance(r["metric"], dict)
                ),
                "retention_count": sum(
                    r["metric"].get("retention") is True
                    for r in projected
                    if isinstance(r["metric"], dict)
                ),
                "current_hardware_execution_count": sum(
                    r["metric"].get("current_hardware_execution") is True
                    for r in projected
                    if isinstance(r["metric"], dict)
                ),
                "new_level_solve_count": sum(
                    r["metric"].get("new_level_solve") is True
                    for r in projected
                    if isinstance(r["metric"], dict)
                ),
                "claim_eligible": bool(
                    task["eligible"] and task["upstream_id"] in {f"Exp{n}" for n in SCIENCE}
                ),
            }
        )
    return units, comparisons


def build_candidate(root: Path, output_root: Path, manifest: Path, date: str) -> dict[str, Any]:
    """Build a private, blocked candidate from current bytes before validation."""
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    tasks, failures, sources = inspect_sources(root)
    units, comparisons = _primitive_rows(root, tasks)
    authority = root / AUTHORITY
    hashes = [
        {
            "path": str(authority),
            "sha256": sha256_file(authority),
            "date": date,
            "role": "active_authority",
            "exposure_status": "current",
        },
        *sources,
    ]
    qualified = {row["upstream_id"] for row in tasks if row["eligible"]}
    required = {f"Exp{n}" for n in SCIENCE}
    complete = required <= qualified and all(row["eligible"] for row in tasks)
    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "experiment_id": 7877,
        "task_id": "exp7877-independent-audit",
        "milestone": "2026.09.683",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v683_science"
        if not complete
        else "complete_null_v683_audit",
        "verdict_class": "blocked" if not complete else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": units,
        "sample_size_budget": {
            "intended": len(units),
            "eligible": sum(r["eligible"] for r in units),
            "started": sum(r["started"] for r in units),
            "completed": sum(r["completed"] for r in units),
            "censored": sum(r["censored"] for r in units),
            "excluded": sum(r["excluded"] for r in units),
            "independent": sum(r["independent"] for r in units),
        },
        "acceptance_gate_results": {
            "validity": int(not failures),
            "readiness": int(complete),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": elapsed,
        "phase_spans": [
            {
                "phase": "preconditions_and_raw_reduction",
                "duration_s": elapsed,
                "completed_units": 12,
            }
        ],
        "random_seed": 7877,
        "source_artifact_hashes": hashes,
        "preconditions_checked": [
            {
                "path": s["path"],
                "hash": s["sha256"],
                "role": s["role"],
                "date": s["date"],
                "schema": _data(Path(s["path"])).get("schema"),
                "passed": s["sha256"] is not None,
            }
            for s in sources
        ],
        "validation_receipts": [],
        "validation_command_manifest_path": str(manifest),
        "observed_child_commands": [],
        "repository_health": {
            "status": "pending",
            "historical_required_failures": [
                f for f in failures if f["artifact_field"].startswith("validation_receipts.")
            ],
        },
        "verifier_is_oracle": False,
        "claim_scope": {
            "fixture": "circular_positive_only",
            "natural_annotations": "exposed_development_only",
            "new_generalization": False,
            "arc": "receipt_delta_only",
            "hardware": "historical_custody_only",
        },
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none",
        "trained_head_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "model_file_hashes": []},
        "audit_execution_ready_score": 0,
        "milestone_evidence_complete_score": int(complete),
        "audit_manifest_path": str(output_root / "audit_manifest.json"),
        "task_evidence_rows": tasks,
        "recomputed_comparison_rows": comparisons,
        "unresolved_gaps": failures,
        "resolved_imports": {
            "carnot.reporting.v683_independent_audit": str(Path(__file__).resolve())
        },
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "sources": hashes,
            "code": sha256_file(Path(__file__)),
            "manifest": sha256_file(manifest),
            "seed": 7877,
        }
    )
    result["current_work_receipt"] = build_current_work_receipt(
        run_id=f"exp7877-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_details={"scope": "current_cpu_audit"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=result["phase_spans"],
    )
    result["field_principles"] = {
        key: "Preserve current evidence and separate audit readiness from scientific benefit."
        for key in result
    }
    result["field_principles"]["acceptance_gate_results"] = {
        key: "Keep unmeasured benefit null and qualification separate."
        for key in result["acceptance_gate_results"]
    }
    return result


def cold_replay(path: Path, root: Path) -> list[str]:
    """Recompute source identities, primitive reductions, and closed log hashes."""
    result = _data(path)
    if not result:
        return ["candidate_unreadable"]
    errors: list[str] = []
    if str(result.get("honest_verdict", "")).startswith("partial_"):
        errors.append("partial_terminal_verdict")
    for source in result.get("source_artifact_hashes", []):
        candidate = Path(source["path"])
        if (sha256_file(candidate) if candidate.is_file() else None) != source["sha256"]:
            errors.append("source_bytes_changed")
    tasks, failures, _ = inspect_sources(root)
    if tasks != result.get("task_evidence_rows"):
        errors.append("task_rows_changed")
    if failures != result.get("gate_check_summary", [])[: len(failures)]:
        errors.append("gate_operands_changed")
    units, comparisons = _primitive_rows(root, tasks)
    if units != result.get("rows") or comparisons != result.get("recomputed_comparison_rows"):
        errors.append("raw_reduction_changed")
    for receipt in result.get("validation_receipts", []):
        log = Path(receipt["log_path"])
        if not log.is_file() or sha256_file(log) != receipt["log_sha256"]:
            errors.append("validation_log_changed")
    return sorted(set(errors))
