"""Read V682 evidence from exact producer bytes (REQ-REPORT-7863)."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file


AUTHORITY = "docs/research-notes/v682-authority-snapshots/active.yaml"
MANIFEST = "/tmp/carnot-7863/validation_command_manifest.json"
SCORES = {
    7852: ("source_boundary_ready_score",),
    7853: ("natural_training_ready_score", "natural_online_ready_score"),
    7854: ("intervention_protocol_ready_score",),
    7855: ("energy_fit_ready_score",),
    7856: ("decision_evidence_ready_score",),
    7857: ("qwen_evidence_ready_score",),
    7858: ("learning_evidence_ready_score",),
    7859: ("feedback_evidence_ready_score",),
    7860: ("arc_delta_ready_score",),
    7861: ("service_evidence_ready_score",),
    7862: ("hardware_evidence_ready_score",),
}


def operand(
    number: int, path: Path, field: str, expected: Any, observed: Any, op: str = "=="
) -> dict[str, Any]:
    """Keep absent operands separate from values that failed a comparison."""
    return {
        "upstream_id": f"Exp{number}",
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }


def check_rows(rows: list[dict[str, Any]], intended: int) -> list[str]:
    """Reject private identifiers and chronology errors before reducing rows."""
    failures: list[str] = []
    if not rows:
        failures.append("zero_rows")
    if len(rows) < intended:
        failures.append("lost_rows")
    keys = [(r.get("family_id"), r.get("arm")) for r in rows]
    if len(keys) != len(set(keys)):
        failures.append("seed_pseudoreplication")
    for row in rows:
        features = " ".join(str(k).lower() for k in row.get("features", {}))
        if any(
            k in features
            for k in ("fixture_id", "family_id", "gold", "label", "annotation", "target")
        ):
            failures.append("fixture_id_leakage")
        if row.get("source_erased") and "source" in features:
            failures.append("source_leakage")
        if (
            row.get("feedback_step") is not None
            and row.get("prediction_step", 0) >= row["feedback_step"]
        ):
            failures.append("future_feedback")
        if row.get("status") == "dropped" and row.get("feedback_replayed"):
            failures.append("dropped_feedback_replayed")
        if row.get("threshold_fit_role") == "evaluation":
            failures.append("evaluation_tuned_threshold")
        if "control_value" in row and row.get("control_value") == row.get("treatment_value"):
            failures.append("identical_controls")
    return sorted(set(failures))


def _required_failures(number: int, path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    """A producer's failed required child cannot be erased by its headline."""
    raw = data.get("validation_receipts", [])
    receipts = raw.get("checks", []) if isinstance(raw, dict) else raw
    failures = []
    for item in receipts if isinstance(receipts, list) else []:
        if not isinstance(item, dict):
            continue
        kind = item.get("classification", item.get("class", item.get("scope")))
        if kind != "required":
            continue
        exit_code = item.get("exit_code")
        if exit_code != 0 or item.get("passed") is False or item.get("timed_out"):
            failures.append(
                operand(
                    number, path, f"validation_receipts.{item.get('name', 'unnamed')}", 0, exit_code
                )
            )
    return failures


def load_data(path: Path) -> dict[str, Any]:
    """Treat malformed external evidence as unavailable without losing its byte hash."""
    try:
        payload = json.loads(path.read_bytes())
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def inspect_sources(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read every declared producer and keep aliases as explanation only."""
    authority = yaml.safe_load((root / AUTHORITY).read_text())
    tasks = authority["tasks"][1:12]
    if [t["id"].split("-", 1)[0] for t in tasks] != [f"exp{n}" for n in range(7852, 7863)]:
        raise ValueError("authority_roster_changed")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for number, task in zip(range(7852, 7863), tasks, strict=True):
        path = root / task["deliverable"]
        aliases = [
            p for p in sorted((root / "results").glob(f"experiment_{number}_*.json")) if p != path
        ]
        data = load_data(path)
        state = (
            "missing"
            if not path.is_file()
            else "invalid_json"
            if not data
            else str(data.get("verdict_class") or "pre_gate_skip")
        )
        current = []
        if not data:
            current.append(
                operand(number, path, "science_producer", "current qualified artifact", state)
            )
        else:
            for field, expected in (
                ("experiment_id", number),
                ("task_id", task["id"]),
                ("milestone", "2026.09.682"),
                ("run_date", "20260929"),
                ("flagged_adversarial", False),
            ):
                if data.get(field) != expected:
                    current.append(operand(number, path, field, expected, data.get(field)))
            if data.get("verdict_class") not in ("positive", "circular_positive", "null"):
                current.append(
                    operand(
                        number,
                        path,
                        "verdict_class",
                        ["positive", "circular_positive", "null"],
                        data.get("verdict_class"),
                        "in",
                    )
                )
            for score in SCORES[number]:
                if data.get(score) != 1:
                    current.append(operand(number, path, score, 1, data.get(score)))
            current.extend(_required_failures(number, path, data))
        failures.extend(current)
        eligible = not current
        if eligible:
            state = "eligible_null" if data.get("verdict_class") == "null" else "eligible"
        source = {
            "upstream_id": f"Exp{number}",
            "task_id": task["id"],
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": data.get("run_date"),
            "role": "science_producer",
            "exposure_status": "exposed_development",
            "status": state,
            "eligibility": eligible,
            "aliases": [
                {"path": str(p), "sha256": sha256_file(p), "role": "explanation_only"}
                for p in aliases
            ],
        }
        sources.append(source)
        budget = data.get("sample_size_budget", {})
        rows.append(
            {
                "upstream_id": f"Exp{number}",
                "path": str(path),
                "sha256": source["sha256"],
                "arm": task["id"],
                "source_family": None,
                "seed": None,
                "status": state,
                "intended": budget.get("intended"),
                "eligible": int(eligible),
                "started": budget.get("started", 0),
                "completed": budget.get("completed", 0),
                "censored": budget.get("censored", 0),
                "excluded": budget.get("excluded", int(not eligible)),
                "independent_n": budget.get("independent_n", budget.get("independent", 0)),
            }
        )
    return rows, failures, sources


def reduce_primitives(
    root: Path, sources: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Count observed rows directly; only labeled predictions support losses."""
    reduced = []
    discrepancies = []
    for source in sources:
        path = Path(source["path"])
        data = load_data(path)
        raw = data.get("rows", []) if isinstance(data, dict) else []
        raw = raw if isinstance(raw, list) else []
        statuses: dict[str, int] = {}
        families: set[str] = set()
        losses = []
        for index, row in enumerate(raw):
            if not isinstance(row, dict):
                discrepancies.append(
                    {
                        "upstream_id": source["upstream_id"],
                        "row_key": index,
                        "artifact_field": "rows",
                        "expected": "object",
                        "observed": type(row).__name__,
                    }
                )
                continue
            status = str(row.get("status", "missing_status"))
            statuses[status] = statuses.get(status, 0) + 1
            family = row.get("source_family", row.get("family_id"))
            if family is not None:
                families.add(str(family))
            probability = row.get("probability")
            label = row.get("label")
            if isinstance(probability, (float, int)) and label in (0, 1):
                losses.append((float(probability) - label) ** 2)
            if row.get("source_erased") and any(
                "source" in str(k).lower() for k in row.get("features", {})
            ):
                discrepancies.append(
                    {
                        "upstream_id": source["upstream_id"],
                        "row_key": row.get("family_id", index),
                        "artifact_field": "features",
                        "expected": "source erased",
                        "observed": list(row["features"]),
                    }
                )
        budget = data.get("sample_size_budget", {}) if isinstance(data, dict) else {}
        intended = budget.get("intended")
        if isinstance(intended, int) and raw and len(raw) != intended:
            discrepancies.append(
                {
                    "upstream_id": source["upstream_id"],
                    "row_key": "all",
                    "artifact_field": "sample_size_budget.intended",
                    "expected": intended,
                    "observed": len(raw),
                }
            )
        reduced.append(
            {
                "upstream_id": source["upstream_id"],
                "path": str(path),
                "sha256": source["sha256"],
                "row_count": len(raw),
                "status_counts": statuses,
                "independent_family_count": len(families),
                "brier": sum(losses) / len(losses) if losses else None,
                "labeled_prediction_count": len(losses),
                "eligible": source["eligibility"],
            }
        )
    return reduced, discrepancies


def primitive_unit_rows(root: Path, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep each observed unit, including excluded and unstarted outcomes."""
    units = []
    for source in sources:
        path = Path(source["path"])
        data = load_data(path)
        raw = data.get("rows", []) if isinstance(data, dict) else []
        raw = raw if isinstance(raw, list) else []
        if not raw:
            units.append(
                {
                    "upstream_id": source["upstream_id"],
                    "path": str(path),
                    "arm": source["task_id"],
                    "source_family": None,
                    "family_id": None,
                    "seed": None,
                    "status": source["status"],
                    "intended": 1,
                    "eligible": 0,
                    "started": 0,
                    "completed": 0,
                    "censored": 0,
                    "excluded": 1,
                    "independent": 0,
                    "metric": None,
                }
            )
            continue
        seen: set[str] = set()
        for index, row in enumerate(raw):
            if not isinstance(row, dict):
                units.append(
                    {
                        "upstream_id": source["upstream_id"],
                        "path": str(path),
                        "arm": None,
                        "source_family": None,
                        "family_id": None,
                        "seed": None,
                        "status": "malformed",
                        "primitive_key": index,
                        "intended": 1,
                        "eligible": 0,
                        "started": 1,
                        "completed": 0,
                        "censored": 0,
                        "excluded": 1,
                        "independent": 0,
                        "metric": None,
                    }
                )
                continue
            family = row.get("source_family", row.get("family_id"))
            key = str(family)
            independent = int(family is not None and key not in seen)
            seen.add(key)
            status = str(row.get("status", "missing_status"))
            units.append(
                {
                    "upstream_id": source["upstream_id"],
                    "path": str(path),
                    "arm": row.get("arm"),
                    "source_family": family,
                    "family_id": row.get("family_id"),
                    "seed": row.get("seed"),
                    "status": status,
                    "primitive_key": index,
                    "intended": 1,
                    "eligible": int(source["eligibility"] and bool(row.get("eligible", True))),
                    "started": int(
                        bool(row.get("started", status not in ("unstarted", "excluded")))
                    ),
                    "completed": int(bool(row.get("completed", status == "completed"))),
                    "censored": int(bool(row.get("censored", status == "censored"))),
                    "excluded": int(bool(row.get("excluded", status == "excluded"))),
                    "independent": independent,
                    "metric": {"probability": row.get("probability"), "label": row.get("label")},
                }
            )
    return units


def mutation_results() -> list[dict[str, Any]]:
    """Record that each private attack is detected without editing evidence."""
    clean = {
        "family_id": "f",
        "source_family": "s",
        "arm": "treatment",
        "seed": 1,
        "status": "completed",
        "features": {"length": 1},
        "prediction_step": 1,
        "feedback_step": 2,
    }
    cases = [
        ("zero_rows", [], 1, "zero_rows"),
        ("lost_rows", [clean], 2, "lost_rows"),
        ("duplicate_seed", [clean, {**clean, "seed": 2}], 2, "seed_pseudoreplication"),
        ("fixture_id", [{**clean, "features": {"fixture_id": "f"}}], 1, "fixture_id_leakage"),
        (
            "source_leakage",
            [{**clean, "source_erased": True, "features": {"source_text": "x"}}],
            1,
            "source_leakage",
        ),
        ("future_feedback", [{**clean, "feedback_step": 1}], 1, "future_feedback"),
        (
            "queue_drop",
            [{**clean, "status": "dropped", "feedback_replayed": True}],
            1,
            "dropped_feedback_replayed",
        ),
        (
            "evaluation_tune",
            [{**clean, "threshold_fit_role": "evaluation"}],
            1,
            "evaluation_tuned_threshold",
        ),
        (
            "identical_controls",
            [{**clean, "control_value": 1, "treatment_value": 1}],
            1,
            "identical_controls",
        ),
    ]
    return [
        {
            "name": name,
            "rejected": expected in check_rows(rows, intended),
            "observed": check_rows(rows, intended),
        }
        for name, rows, intended, expected in cases
    ]


def precondition_record(source: dict[str, Any]) -> dict[str, Any]:
    """Record the observed schema and ownership even when source JSON is bad."""
    schema = None
    if source["sha256"] is not None:
        try:
            payload = json.loads(Path(source["path"]).read_bytes())
            schema = payload.get("schema") if isinstance(payload, dict) else None
        except (OSError, ValueError, UnicodeError):
            pass
    return {
        "upstream_id": source["upstream_id"],
        "path": source["path"],
        "hash": source["sha256"],
        "artifact_field": "science_producer",
        "op": "exists",
        "expected": True,
        "observed": source["sha256"] is not None,
        "schema_version": schema,
        "source_role": source["role"],
        "exposure_status": source["exposure_status"],
        "resource_ownership": "external_producer",
        "passed": source["eligibility"],
    }


def build_candidate(root: Path, output_root: Path, date: str) -> dict[str, Any]:
    """Create a complete blocked reading before child validation starts."""
    started = time.monotonic()
    status_rows, failures, sources = inspect_sources(root)
    reductions, discrepancies = reduce_primitives(root, sources)
    authority = root / AUTHORITY
    source_hashes = [
        {
            "path": str(authority),
            "sha256": sha256_file(authority),
            "role": "active_authority",
            "date": "20260929",
            "exposure_status": "current",
        }
    ]
    for source in sources:
        source_hashes.append(
            {
                "path": source["path"],
                "sha256": source["sha256"],
                "role": "science_producer",
                "date": source["date"],
                "exposure_status": source["exposure_status"],
            }
        )
        source_hashes.extend(
            {**alias, "date": None, "exposure_status": "historical_or_administrative"}
            for alias in source["aliases"]
        )
    historical = json.loads(
        (root / "results/experiment_7849_v681_independent_audit.json").read_bytes()
    )
    old_failures = [
        r
        for r in historical.get("validation_receipts", [])
        if r.get("classification") == "required" and not r.get("passed")
    ]
    old_failures.extend(
        historical.get("repository_health", {}).get("historical_required_failures", [])
    )
    old_health = historical.get("repository_health", {}).get("receipt")
    if isinstance(old_health, dict) and not old_health.get("passed"):
        old_failures.append({**old_health, "classification": "historical_diagnostic"})
    manifest = Path(MANIFEST)
    elapsed = time.monotonic() - started
    eligible = sum(s["eligibility"] for s in sources)
    result: dict[str, Any] = {
        "experiment_id": 7863,
        "task_id": "exp7863-independent-audit",
        "milestone": "2026.09.682",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v682_science",
        "verdict_class": "blocked",
        "flagged_adversarial": bool(discrepancies and eligible == 11),
        "gate_check_summary": failures,
        "rows": primitive_unit_rows(root, sources),
        "sample_size_budget": {
            "intended": 11,
            "eligible": eligible,
            "started": sum(s["sha256"] is not None for s in sources),
            "completed": eligible,
            "censored": 0,
            "excluded": 11 - eligible,
            "independent_n": sum(
                r["independent_family_count"] for r in reductions if r["eligible"]
            ),
        },
        "acceptance_gate_results": {
            "validity": not failures,
            "readiness": int(eligible == 11),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": elapsed,
        "phase_spans": [
            {"phase": "preconditions_and_reduction", "duration_s": elapsed, "completed_units": 11}
        ],
        "random_seed": 7863,
        "reproducibility_checksum": canonical_hash(
            {
                "sources": source_hashes,
                "manifest": sha256_file(manifest),
                "code": sha256_file(Path(__file__)),
                "seed": 7863,
            }
        ),
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": [precondition_record(source) for source in sources],
        "validation_receipts": [],
        "validation_command_manifest_path": str(manifest),
        "observed_child_commands": [],
        "repository_health": {"status": "pending", "historical_required_failures": old_failures},
        "verifier_is_oracle": False,
        "claim_scope": {
            "fixtures": "circular_positive_only",
            "natural_annotations": "exposed_development_only",
            "fresh_generalization_eligible": False,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "model_file_hashes": []},
        "audit_execution_ready_score": 0,
        "eligible_producer_count": eligible,
        "producer_status_rows": status_rows,
        "independently_reduced_rows": reductions,
        "discrepancy_rows": discrepancies,
        "mutation_results": mutation_results(),
        "milestone_evidence_complete_score": int(eligible == 11),
        "resolved_imports": {
            "carnot.reporting.v682_independent_audit": str(Path(__file__).resolve())
        },
        "output_root": str(output_root),
    }
    result["field_principles"] = {
        key: "Preserve exact V682 provenance, observed operands, and claim scope." for key in result
    }
    result["field_principles"]["acceptance_gate_results"] = {
        key: "Keep validity and scientific benefit separate; unmeasured gates stay null."
        for key in result["acceptance_gate_results"]
    }
    return result


def cold_replay(path: Path, root: Path) -> list[str]:
    """Re-read all source and closed-log bytes after the candidate is written."""
    try:
        result = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return ["candidate_unreadable"]
    errors = []
    for key, expected in (
        ("experiment_id", 7863),
        ("task_id", "exp7863-independent-audit"),
        ("milestone", "2026.09.682"),
        ("run_date", "20260929"),
    ):
        if result.get(key) != expected:
            errors.append(key)
    if str(result.get("honest_verdict", "")).startswith("partial_"):
        errors.append("partial_terminal_verdict")
    for source in result.get("source_artifact_hashes", []):
        source_path = Path(source["path"])
        observed = sha256_file(source_path) if source_path.is_file() else None
        if observed != source.get("sha256"):
            errors.append("source_bytes_changed")
    rows, failures, sources = inspect_sources(root)
    if primitive_unit_rows(root, sources) != result.get("rows") or rows != result.get(
        "producer_status_rows"
    ):
        errors.append("rows_changed")
    if failures != result.get("gate_check_summary", [])[: len(failures)]:
        errors.append("gate_operands_changed")
    reduced, discrepancies = reduce_primitives(root, sources)
    if reduced != result.get("independently_reduced_rows") or discrepancies != result.get(
        "discrepancy_rows"
    ):
        errors.append("primitive_reduction_changed")
    for receipt in result.get("validation_receipts", []):
        log_path = Path(receipt["log_path"])
        if not log_path.is_file() or sha256_file(log_path) != receipt.get("log_sha256"):
            errors.append("validation_log_changed")
    return sorted(set(errors))
