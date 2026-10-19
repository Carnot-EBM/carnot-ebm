"""Reduce V685 producer bytes without trusting producer summary metrics.

REQ-REPORT-7902-V685. Administrative completion records what was checked;
scientific benefit requires qualified rows from an independent source.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v685_authority_lifecycle import assess_authorities


QUALIFIED = {"positive", "circular_positive", "null"}
SCIENCE = {7892, 7893, 7894, 7895, 7896, 7897, 7898, 7900}
METRICS = (
    "probability",
    "label",
    "cost",
    "false_accept",
    "latency_ms",
    "feedback_step",
    "prediction_step",
    "bank_write_step",
    "retention",
)


def _read(path: Path) -> dict[str, Any] | None:
    """An unreadable producer is absent evidence, even if a path exists."""
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def _operand(
    number: int, path: Path, field: str, expected: Any, observed: Any, op: str = "=="
) -> dict[str, Any]:
    """Name the failed value and exact bytes to prevent silent gate changes."""
    return {
        "upstream_id": f"Exp{number}",
        "artifact_path": str(path),
        "artifact_hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count families once and recompute scores from completed observations."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    counts = {
        key: 0
        for key in (
            "intended",
            "eligible",
            "started",
            "completed",
            "failed",
            "censored",
            "excluded",
        )
    }
    families: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed_primitive_row")
        counts["intended"] += 1
        status = str(row.get("status", "completed" if row.get("completed") is True else "unknown"))
        excluded = status == "excluded" or row.get("excluded") is True
        counts["excluded"] += int(excluded)
        counts["eligible"] += int(not excluded)
        counts["started"] += int(status not in {"unstarted", "excluded"})
        counts["completed"] += int(status == "completed" or row.get("completed") is True)
        counts["failed"] += int(status == "failed")
        counts["censored"] += int(status == "censored" or row.get("censored") is True)
        family = row.get("family_id", row.get("family"))
        if family is not None and not excluded:
            families.add(str(family))
        if status == "completed" or row.get("completed") is True:
            if family is not None:
                grouped[(str(family), str(row.get("arm", "default")))].append(row)
    per_arm: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_family: dict[tuple[str, str], dict[str, float]] = {}
    for key, repeats in sorted(grouped.items()):
        labels = {r["label"] for r in repeats if r.get("label") in (0, 1)}
        if len(labels) > 1:
            raise ValueError("family_label_conflict")
        if any(
            "original_human_label" in r and r.get("label") != r["original_human_label"]
            for r in repeats
        ):
            raise ValueError("human_label_changed")
        label = next(iter(labels)) if labels else None
        values: dict[str, float] = {}
        for metric in ("probability", "cost"):
            measured = [float(r[metric]) for r in repeats if type(r.get(metric)) in (float, int)]
            if measured:
                values[metric] = sum(measured) / len(measured)
                per_arm[key[1]][metric].append(values[metric])
        if label is not None:
            risks = [
                (float(r["probability"]) - label) ** 2
                for r in repeats
                if type(r.get("probability")) in (float, int)
            ]
            if risks:
                values["brier"] = sum(risks) / len(risks)
                per_arm[key[1]]["brier"].append(values["brier"])
        false_accepts = sum(
            r.get("false_accept") is True or (r.get("action") == "accept" and r.get("label") == 0)
            for r in repeats
        )
        values["false_accepts"] = false_accepts / len(repeats)
        per_arm[key[1]]["false_accepts"].append(values["false_accepts"])
        per_family[key] = values
    summary: dict[str, Any] = {
        **counts,
        "independent_families": len(families),
        "cluster_denominators": {
            arm: len(vals.get("false_accepts", [])) for arm, vals in sorted(per_arm.items())
        },
    }
    for metric, target in (
        ("probability", "probability_by_arm"),
        ("brier", "brier_by_arm"),
        ("cost", "cost_by_arm"),
        ("false_accepts", "false_accepts_by_arm"),
    ):
        summary[target] = {
            arm: sum(vals[metric]) / len(vals[metric])
            for arm, vals in sorted(per_arm.items())
            if vals.get(metric)
        }
    arms = sorted(per_arm, key=lambda arm: ("control" in arm or arm == "no_write", arm))
    summary["paired_cost_ci95"] = None
    if len(arms) == 2:
        pairs = [
            per_family[(family, arms[1])]["cost"] - per_family[(family, arms[0])]["cost"]
            for family in sorted(families)
            if (family, arms[0]) in per_family
            and (family, arms[1]) in per_family
            and "cost" in per_family[(family, arms[0])]
            and "cost" in per_family[(family, arms[1])]
        ]
        if pairs:
            rng = random.Random(7902)
            draws = sorted(sum(rng.choice(pairs) for _ in pairs) / len(pairs) for _ in range(1000))
            summary["paired_cost_ci95"] = [draws[24], draws[974]]
    return summary


def _science_audit(number: int, rows: list[dict[str, Any]]) -> list[str]:
    """Check event order and calibration claims without using producer reducers."""
    errors: list[str] = []
    if number == 7897:
        for row in rows:
            prediction = row.get("prediction_step")
            feedback = row.get("feedback_step")
            write = row.get("bank_write_step")
            later = row.get("later_decision_step")
            if all(type(x) is int for x in (prediction, feedback, write, later)):
                if not prediction < feedback <= write < later:
                    errors.append("feedback_bank_decision_order")
    if number == 7898:
        fit = {str(r.get("family_id")) for r in rows if r.get("buffer_role") == "fit"}
        test = {str(r.get("family_id")) for r in rows if r.get("buffer_role") == "test"}
        if fit & test:
            errors.append("calibration_buffer_overlap")
        pairs = [
            (float(r["raw_confidence"]), float(r["transformed_confidence"]))
            for r in rows
            if type(r.get("raw_confidence")) in (int, float)
            and type(r.get("transformed_confidence")) in (int, float)
        ]
        if any(
            right[1] < left[1]
            for left, right in zip(sorted(pairs), sorted(pairs)[1:])
            if right[0] > left[0]
        ):
            errors.append("nonmonotone_confidence_transform")
    return sorted(set(errors))


def build_candidate(
    root: Path, design: Path, active: Path, date: str, publication: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a byte-bound twelve-task diagnosis from an activated authority."""
    started = time.monotonic()
    if date != "20260929":
        raise ValueError("v685_date_changed")
    authority = assess_authorities(
        design,
        root / "research-roadmap-next.yaml",
        active,
        root / "results/raw/experiment_7902_v685_capstone/authority",
    )
    if not authority["activated"]:
        raise ValueError("authority_contract_changed")
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    if len(tasks) != 12 or [t["id"].split("-", 1)[0] for t in tasks] != [
        f"exp{number}" for number in range(7891, 7903)
    ]:
        raise ValueError("authority_task_order_changed")
    source_hashes = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "role": role,
            "exposure_status": "administrative",
        }
        for path, role in ((design, "design_contract"), (active, "active_authority"))
    ]
    outcomes: list[dict[str, Any]] = []
    reductions: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    historical: list[dict[str, Any]] = []
    observed: dict[str, dict[str, Any] | None] = {}
    for index, task in enumerate(tasks):
        number = 7891 + index
        path = root / task["deliverable"]
        if number == 7902:
            data = None
            status = "self_administrative"
        else:
            data = _read(path)
            status = "missing" if data is None else str(data.get("verdict_class", "inconsistent"))
        observed[task["id"]] = data
        digest = sha256_file(path) if path.is_file() and number != 7902 else None
        source_hashes.append(
            {
                "path": str(path),
                "sha256": digest,
                "role": "self_administrative"
                if number == 7902
                else "scientific_producer"
                if number in SCIENCE
                else "continuity_producer",
                "exposure_status": "exposed_development" if number in SCIENCE else "administrative",
            }
        )
        if number != 7902 and data is None:
            failures.append(_operand(number, path, "producer_exists", True, path.is_file()))
        if data is not None:
            for field, expected in (
                ("experiment_id", number),
                ("task_id", task["id"]),
                ("milestone", "2026.09.685"),
                ("run_date", date),
                ("MODEL_SPECS", task["MODEL_SPECS"]),
                ("flagged_adversarial", False),
            ):
                actual = data.get(field, "missing_field")
                if actual != expected:
                    failures.append(_operand(number, path, field, expected, actual))
            if data.get("verdict_class") not in QUALIFIED:
                failures.append(
                    _operand(
                        number,
                        path,
                        "verdict_class",
                        sorted(QUALIFIED),
                        data.get("verdict_class", "missing_field"),
                        "in",
                    )
                )
            receipts = data.get("validation_receipts", [])
            checks = receipts.get("checks", []) if isinstance(receipts, dict) else receipts
            for check in checks if isinstance(checks, list) else []:
                if (
                    isinstance(check, dict)
                    and check.get("classification", check.get("scope")) == "required"
                    and check.get("passed") is False
                ):
                    historical.append(
                        {
                            "upstream_id": f"Exp{number}",
                            "path": str(path),
                            "name": check.get("name"),
                            "exit_code": check.get("exit_code"),
                            "log_path": check.get("log_path"),
                            "log_sha256": check.get("log_sha256"),
                        }
                    )
                    failures.append(
                        _operand(
                            number, path, f"validation_receipts.{check.get('name')}", True, False
                        )
                    )
        if number != 7902:
            for gate in task["gated_on"]:
                upstream = next(t for t in tasks[:index] if t["id"] == gate["upstream"])
                upstream_number = int(upstream["id"][3:7])
                upstream_path = root / upstream["deliverable"]
                prior = observed[upstream["id"]]
                actual = (
                    prior.get(gate["artifact_field"], "missing_field")
                    if prior
                    else "missing_source"
                )
                passed = (
                    actual == gate["value"]
                    if gate["op"] == "=="
                    else actual in gate["value"]
                    if gate["op"] == "in"
                    else False
                )
                if not passed:
                    failures.append(
                        _operand(
                            upstream_number,
                            upstream_path,
                            gate["artifact_field"],
                            gate["value"],
                            actual,
                            gate["op"],
                        )
                    )
        own_errors = [
            f
            for f in failures
            if f["upstream_id"] == f"Exp{number}" and f["artifact_path"] == str(path)
        ]
        if data is not None and own_errors and status in QUALIFIED:
            status = "inconsistent"
        raw = data.get("rows", []) if data else []
        if not isinstance(raw, list):
            failures.append(_operand(number, path, "rows", "list", type(raw).__name__))
            raw = []
        try:
            reduced = reduce_rows(raw)
            audit_errors = _science_audit(number, raw)
        except ValueError as error:
            reduced = reduce_rows([])
            audit_errors = [str(error)]
        for error in audit_errors:
            failures.append(_operand(number, path, error, False, True))
            status = "inconsistent"
        reductions.append(
            {
                "upstream_id": f"Exp{number}",
                "path": str(path),
                "hash": digest,
                **reduced,
                "audit_errors": audit_errors,
            }
        )
        if not raw:
            units.append(
                {
                    "upstream_id": f"Exp{number}",
                    "family_id": None,
                    "arm": task["id"],
                    "seed": None,
                    "status": status,
                    "intended": 1,
                    "eligible": 0,
                    "started": 0,
                    "completed": 0,
                    "failed": 0,
                    "censored": 0,
                    "excluded": int(number != 7902),
                    "independent": 0,
                    "metric": {},
                }
            )
        for position, row in enumerate(raw):
            row = row if isinstance(row, dict) else {}
            state = row.get("status", "unknown")
            row_eligible = (
                status in QUALIFIED and not own_errors and not audit_errors and state != "excluded"
            )
            units.append(
                {
                    "upstream_id": f"Exp{number}",
                    "primitive_index": position,
                    "family_id": row.get("family_id", row.get("family")),
                    "arm": row.get("arm"),
                    "seed": row.get("seed"),
                    "status": state,
                    "intended": 1,
                    "eligible": int(row_eligible),
                    "started": int(state not in ("unstarted", "excluded")),
                    "completed": int(state == "completed" or row.get("completed") is True),
                    "failed": int(state == "failed"),
                    "censored": int(state == "censored"),
                    "excluded": int(not row_eligible),
                    "independent": 0,
                    "metric": {k: row[k] for k in METRICS if k in row},
                }
            )
        outcomes.append(
            {
                "upstream_id": f"Exp{number}",
                "task_id": task["id"],
                "path": str(path),
                "hash": digest,
                "status": status,
                "eligible": status in QUALIFIED and not own_errors and not audit_errors,
                "role": "self_administrative"
                if number == 7902
                else "scientific_producer"
                if number in SCIENCE
                else "continuity_producer",
                "source_exposure": "exposed_development" if number in SCIENCE else "administrative",
            }
        )
    seen: set[tuple[str, str]] = set()
    for unit in units:
        family = unit.get("family_id")
        key = (unit["upstream_id"], str(family))
        if family is not None and unit["eligible"] and key not in seen:
            unit["independent"] = 1
            seen.add(key)
    budget = {
        key: sum(int(unit[key]) for unit in units)
        for key in (
            "intended",
            "eligible",
            "started",
            "completed",
            "failed",
            "censored",
            "excluded",
            "independent",
        )
    }
    science_ready = all(
        row["eligible"] for row in outcomes if int(row["upstream_id"][3:]) in SCIENCE
    )
    verdict = (
        "complete_null_independent_benefit_unshown"
        if science_ready
        else "complete_blocked_missing_science"
    )
    verdict_class = "null" if science_ready else "blocked"
    publication = publication or {
        "G1": False,
        "G2": False,
        "G3": False,
        "G4": False,
        "paper_ready": False,
        "unmet_gates": ["G1", "G2", "G3", "G4"],
    }
    gates = {
        key: bool(publication.get("gates", {}).get(key, {}).get("pass", publication.get(key)))
        for key in ("G1", "G2", "G3", "G4")
    }
    gap_members = {
        "FR-12": (7892, 7893, 7894, 7895, 7896),
        "FR-11": (7897, 7898),
        "FR-05/FR-08/NFR-01": (7899, 7900, 7901),
        "GAP-ORACLE-DISTINCT": (7894, 7895),
    }
    by_number = {int(row["upstream_id"][3:]): row for row in outcomes}
    gap_decisions = {}
    for gap, members in gap_members.items():
        states = [by_number[number]["status"] for number in members]
        decision = (
            "blocked"
            if "missing" in states or "blocked" in states
            else "disqualified"
            if "disqualified" in states or "inconsistent" in states
            else "measured-null"
            if "null" in states
            else "advanced"
        )
        gap_decisions[gap] = {
            "decision": decision,
            "required_producers": list(members),
            "observed_statuses": states,
            "independent_benefit": None if decision != "advanced" else False,
        }
    retirement = [
        {
            "experiment_id": item.get("experiment_id"),
            "prior_verdict": item.get("verdict"),
            "retire_if_same_verdict": item.get("retire_if_same_verdict"),
            "decision": "retained",
            "reason": "No identical current failure scope and verdict was established.",
        }
        for item in tasks[-1].get("prior_failures", [])
    ]
    result: dict[str, Any] = {
        "experiment_id": 7902,
        "task_id": "exp7902-capstone",
        "milestone": "2026.09.685",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": units,
        "sample_size_budget": budget,
        "acceptance_gate_results": {
            "validity": True,
            "readiness": 1,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": [
            {
                "phase": "authority_and_independent_reduction",
                "duration_s": time.monotonic() - started,
                "completed_units": 12,
            }
        ],
        "random_seed": 7902,
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": [
            {
                "path": s["path"],
                "hash": s["sha256"],
                "source_role": s["role"],
                "observed": s["sha256"] is not None,
            }
            for s in source_hashes
        ],
        "resolved_imports": {
            "carnot.reporting.v685_capstone": str(Path(__file__).resolve()),
            "carnot.reporting.v685_authority_lifecycle": str(
                Path(assess_authorities.__code__.co_filename).resolve()
            ),
        },
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "historical_required_failures": historical,
        "repository_health": {
            "status": "historical_diagnostic_open",
            "repository_wide_check_repeated": False,
        },
        "verifier_is_oracle": False,
        "claim_scope": {
            "natural_data": "exposed_development",
            "fixture_agreement": "circular_only",
            "fresh_holdout_generalization": False,
            "gap_oracle_distinct": "open",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none",
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "model_file_hashes": []},
        "trained_head_specs": [],
        "capstone_execution_ready_score": 1,
        "outcome_rows": outcomes,
        "independent_reduction_rows": reductions,
        "gap_decisions": gap_decisions,
        "retirement_decisions": retirement,
        **gates,
        "paper_ready": all(gates.values()),
        "unmet_gates": [key for key, met in gates.items() if not met],
        "publication_gate_results": publication,
        "report_path": "docs/research-notes/experiment_7902_v685_capstone.md",
        "authority_snapshots": authority["authority_snapshots"],
        "canonical_tasks_sha256": authority["canonical_tasks_sha256"],
    }
    result["reproducibility_checksum"] = canonical_hash(
        {"sources": source_hashes, "rows": units, "seed": 7902, "config": "v685_capstone_v1"}
    )
    result["field_principles"] = {
        "source_artifact_hashes": "Bind every input role to its exact bytes.",
        "gate_check_summary": "Keep missing and failed operands distinct.",
        "rows": "Preserve primitive units for a cold reader.",
        "sample_size_budget": "Count families separately from repeated seeds.",
        "acceptance_gate_results": "Separate mechanics from unmeasured benefit.",
        "capstone_execution_ready_score": "A complete diagnosis can be ready while science is blocked.",
        "gap_decisions": "Classify each product gap from qualified evidence only.",
        "paper_ready": "Require the stable G1 through G4 conjunction.",
    }
    return result


def cold_replay(candidate: Path, root: Path, design: Path, active: Path) -> list[str]:
    """Recreate current evidence and reject a changed source or claimed count."""
    value = _read(candidate)
    if value is None:
        return ["candidate_unreadable"]
    expected = build_candidate(
        root, design, active, "20260929", value.get("publication_gate_results")
    )
    errors = []
    for source in value.get("source_artifact_hashes", []):
        if source.get("role") == "self_administrative":
            continue
        path = Path(source["path"])
        digest = sha256_file(path) if path.is_file() else None
        if digest != source.get("sha256"):
            errors.append("source_bytes_changed")
    for field in (
        "outcome_rows",
        "independent_reduction_rows",
        "rows",
        "sample_size_budget",
        "gate_check_summary",
        "reproducibility_checksum",
    ):
        if value.get(field) != expected[field]:
            errors.append(f"{field}_changed")
    return sorted(set(errors))
