"""Reopen V678 source custody and primitive rows (REQ-REPORT-7807)."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

PLAN = {
    7799: "results/experiment_7799_v678_decision_measurement.json",
    7801: "results/experiment_7801_v678_qwen_counter_evidence.json",
    7802: "results/experiment_7802_v678_continuous_acquisition.json",
}
PRE_GATE = {7801: "results/experiment_7801_qwen_counter_evidence.json"}
GATES = (
    "validity",
    "readiness",
    "probability_quality",
    "decision_benefit",
    "retention",
    "efficiency",
)


def failure(
    number: int, path: Path, field: str, expected: Any, observed: Any, operator: str = "=="
) -> dict[str, Any]:
    """Keep both operands and exact file bytes for a failed current gate."""
    return {
        "upstream_id": f"Exp{number}",
        "artifact_path": str(path.resolve()),
        "artifact_hash": sha256_file(path) if path.is_file() else None,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
    }


def inspect_sources(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inspect declared science paths and keep pre-gate receipts distinct."""
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number, relative in PLAN.items():
        path = root / relative
        receipt = root / PRE_GATE[number] if number in PRE_GATE else None
        source: dict[str, Any] = {
            "upstream_id": f"Exp{number}",
            "path": relative,
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": None,
            "imported_fields": {},
            "raw_paths": {},
            "state": "missing",
            "eligibility": False,
            "pre_gate_receipt": None,
        }
        if receipt is not None and receipt.is_file():
            source["pre_gate_receipt"] = {
                "path": str(receipt.relative_to(root)),
                "sha256": sha256_file(receipt),
                "role": "explanation_only",
            }
            try:
                gate = json.loads(receipt.read_bytes())
                for item in gate.get("gates_evaluated", []):
                    if item.get("passed") is False:
                        upstream = root / str(item.get("artifact_path", ""))
                        row = failure(
                            number,
                            upstream,
                            item["artifact_field"],
                            item.get("expected"),
                            item.get("actual"),
                            item.get("op", "=="),
                        )
                        row["gate_upstream"] = item.get("upstream")
                        failures.append(row)
            except (ValueError, KeyError, TypeError):
                failures.append(
                    failure(number, receipt, "pre_gate_schema", "JSON gates_evaluated", "invalid")
                )
        if not path.is_file():
            failures.append(
                failure(
                    number, path, "producer_path", "existing declared science producer", "missing"
                )
            )
            sources.append(source)
            continue
        try:
            value = json.loads(path.read_bytes())
            if not isinstance(value, dict):
                raise ValueError("not object")
        except (ValueError, UnicodeError):
            value = {}
            failures.append(failure(number, path, "schema", "JSON object", "invalid"))
        expected = {
            "experiment_id": number,
            "milestone": "2026.09.678",
            "run_date": "20260928",
            "flagged_adversarial": False,
        }
        for field, wanted in expected.items():
            if value.get(field) != wanted:
                failures.append(failure(number, path, field, wanted, value.get(field)))
        if value.get("verdict_class") not in {"positive", "null", "circular_positive"}:
            failures.append(
                failure(
                    number,
                    path,
                    "verdict_class",
                    "qualified complete science",
                    value.get("verdict_class"),
                    "in",
                )
            )
        if not str(value.get("honest_verdict", "")).startswith("complete_"):
            failures.append(
                failure(number, path, "honest_verdict", "complete_*", value.get("honest_verdict"))
            )
        raw = value.get("raw_rows_path")
        safe = (
            isinstance(raw, str)
            and bool(raw)
            and not Path(raw).is_absolute()
            and ".." not in Path(raw).parts
        )
        raw_path = root / raw if safe else path
        digest = sha256_file(raw_path) if safe and raw_path.is_file() else None
        source["raw_paths"]["raw_rows_path"] = {"path": raw, "sha256": digest}
        if digest is None:
            failures.append(
                failure(number, raw_path, "raw_rows_path", "existing safe declared bytes", raw)
            )
        if value.get("raw_rows_sha256") != digest:
            failures.append(
                failure(number, raw_path, "raw_rows_sha256", value.get("raw_rows_sha256"), digest)
            )
        source["date"] = value.get("run_date")
        source["imported_fields"] = {
            key: value.get(key) for key in (*expected, "verdict_class", "honest_verdict")
        }
        source["state"] = (
            "disqualified"
            if any(f["upstream_id"] == f"Exp{number}" for f in failures)
            else "eligible"
        )
        source["eligibility"] = source["state"] == "eligible"
        sources.append(source)
    return sources, failures


def reduce_fixture(rows: list[dict[str, Any]], expected_families: int) -> dict[str, Any]:
    """Score saved primitive probabilities with each family counted once."""
    errors: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    for row in rows:
        family = row.get("family_id")
        grouped[str(family)].append(row)
        features = row.get("feature_names", [])
        if any("label" in name or "confidence" in name for name in features):
            errors.add("feature_leakage")
        if row.get("label_origin") != "independent_annotation":
            errors.add("self_label")
        if row.get("label_join") != family:
            errors.add("label_join")
        if row.get("prediction_tick", 0) >= row.get("label_tick", 0):
            errors.add("future_feedback")
        probability, label = row.get("probability"), row.get("label")
        if (
            type(probability) not in (int, float)
            or not 0 <= probability <= 1
            or type(label) is not int
            or label not in (0, 1)
        ):
            errors.add("primitive_schema")
            metric = None
        else:
            brier = (probability - label) ** 2
            metric = {
                "brier": brier,
                "cost": 0.25
                if row.get("action") == "escalate"
                else 5.0
                if row.get("action") == "accept" and label == 1
                else 1.0
                if row.get("action") == "reject" and label == 0
                else 0.0,
            }
            if "brier" in row and not math.isclose(row["brier"], brier, abs_tol=1e-9):
                errors.add("saved_metric")
        output.append(
            {
                "family_id": family,
                "arm": row.get("arm"),
                "seed": row.get("seed"),
                "role": row.get("role"),
                "label": label,
                "metrics": metric,
                "raw_provenance": {"source_sha256": row.get("source_sha256")},
                "censored": bool(row.get("censored", False)),
                "exclusions": [] if metric else ["invalid_metric"],
            }
        )
    if len(grouped) != expected_families:
        errors.add("family_roster")
    roster = {(row.get("arm"), row.get("seed")) for row in rows}
    for group in grouped.values():
        if {(row.get("arm"), row.get("seed")) for row in group} != roster or len(group) != len(
            roster
        ):
            errors.add("family_roster")
        if (
            len({row.get("label") for row in group}) != 1
            or len({row.get("source_sha256") for row in group}) != 1
        ):
            errors.add("label_join")
    return {"failed_checks": sorted(errors), "independent_n": len(grouped), "rows": output}


def check_interval_units(rows: list[dict[str, Any]], claimed_n: int) -> list[str]:
    """Reject an interval that treats seeds or arm views as new families."""
    return [] if claimed_n == len({row.get("family_id") for row in rows}) else ["seed_as_sample"]


def check_events(events: list[dict[str, Any]]) -> list[str]:
    """Replay arrival, admission, commit, and restart clocks."""
    errors: set[str] = set()
    predictions: dict[tuple[Any, Any], float] = {}
    admitted: set[tuple[Any, Any]] = set()
    queued: set[tuple[Any, Any]] = set()
    for event in sorted(events, key=lambda row: row.get("tick", -1)):
        kind = event.get("kind")
        key = (event.get("arm"), event.get("family_id"))
        tick = event.get("tick", -1)
        if kind == "prediction":
            predictions[key] = tick
        elif kind == "feedback":
            if key not in predictions or tick <= predictions[key]:
                errors.add("future_feedback")
            queued.add(key)
        elif kind == "admission":
            if key in admitted or key not in queued:
                errors.add("duplicate_or_early_admission")
            admitted.add(key)
        elif kind == "commit":
            if key not in admitted:
                errors.add("unreleased_commit")
            queued.discard(key)
        elif kind == "restart" and set(map(tuple, event.get("queued", []))) != queued:
            errors.add("restart_queue_mismatch")
        if kind == "shuffle" and event.get("source_arm") != event.get("arm"):
            errors.add("cross_arm_shuffle")
    return sorted(errors)


def read_branches(
    root: Path, sources: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Read qualified raw bytes without using a producer metric reducer."""
    branches: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for source in sources:
        if source["state"] != "eligible":
            continue
        number = int(source["upstream_id"][3:])
        raw = root / source["raw_paths"]["raw_rows_path"]["path"]
        try:
            value = json.loads(raw.read_bytes())
            rowsets = value["rowsets"]
            branches[number] = {
                role: reduce_fixture(
                    rows, 32 if role == "retention32" else 48 if number == 7801 else 64
                )
                for role, rows in rowsets.items()
            }
            checks = [
                check for part in branches[number].values() for check in part["failed_checks"]
            ]
            if checks:
                raise ValueError(",".join(sorted(set(checks))))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            source["state"] = "disqualified"
            source["eligibility"] = False
            failures.append(
                failure(
                    number,
                    raw,
                    "raw_reduction",
                    "valid family-level primitive rows",
                    type(exc).__name__ + ":" + str(exc),
                )
            )
    return branches, failures


def private_mutations() -> list[dict[str, Any]]:
    """Challenge the same primitive checker with six private corruptions."""
    import copy

    base = [
        {
            "family_id": family,
            "seed": seed,
            "arm": arm,
            "role": "evaluation64",
            "label": 0,
            "label_join": family,
            "label_origin": "independent_annotation",
            "feature_names": ["public_source_length"],
            "source_sha256": family,
            "prediction_tick": 1,
            "label_tick": 2,
            "probability": 0.2,
            "action": "accept",
            "brier": 0.04,
        }
        for family in ("a", "b")
        for arm in ("candidate", "control")
        for seed in (0, 1)
    ]
    outcomes = []
    for name, field, value, expected in (
        ("leaked_label", "feature_names", ["private_label"], "feature_leakage"),
        ("leaked_confidence", "feature_names", ["gold_confidence"], "feature_leakage"),
        ("copied_self_label", "label_origin", "self_label", "self_label"),
        ("altered_prediction", "probability", 0.8, "saved_metric"),
        ("future_feedback", "prediction_tick", 3, "future_feedback"),
    ):
        changed = copy.deepcopy(base)
        changed[0][field] = value
        observed = reduce_fixture(changed, 2)["failed_checks"]
        outcomes.append(
            {
                "mutation": name,
                "expected_check": expected,
                "observed_checks": observed,
                "rejected": expected in observed,
            }
        )
    dropped = reduce_fixture(base[:-1], 2)["failed_checks"]
    outcomes.append(
        {
            "mutation": "removed_unfavorable_row",
            "expected_check": "family_roster",
            "observed_checks": dropped,
            "rejected": "family_roster" in dropped,
        }
    )
    interval = check_interval_units(base, len(base))
    outcomes.append(
        {
            "mutation": "seed_as_sample_interval",
            "expected_check": "seed_as_sample",
            "observed_checks": interval,
            "rejected": "seed_as_sample" in interval,
        }
    )
    return outcomes


def build_artifact(
    root: Path,
    date: str,
    sources: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    branches: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Keep current branch states and leave unmeasured science null."""
    ready = int(
        len(branches) == 3 and all(s["state"] == "eligible" for s in sources) and not failures
    )
    blocked = any(s["state"] == "missing" for s in sources)
    rows = [
        {
            "upstream_id": s["upstream_id"],
            "state": s["state"],
            "role": "science_source",
            "family_id": None,
            "seed": None,
            "arm": None,
            "metrics": None,
            "raw_provenance": {"path": s["path"], "sha256": s["sha256"]},
            "exclusions": [s["state"]] if s["state"] != "eligible" else [],
            "censored": False,
        }
        for s in sources
    ]
    for number, parts in branches.items():
        for role, reduction in parts.items():
            rows.extend({**row, "upstream_id": f"Exp{number}"} for row in reduction["rows"])
    artifact: dict[str, Any] = {
        "schema": "independent_evidence_audit_v1",
        "experiment_id": 7807,
        "milestone": "2026.09.678",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v678_evidence"
        if blocked
        else "complete_disqualified_source_custody"
        if failures
        else "complete_null_exposed_development",
        "verdict_class": "blocked" if blocked else "disqualified" if failures else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "discrepancy_rows": failures,
        "branch_dispositions": [
            {
                "upstream_id": s["upstream_id"],
                "state": s["state"],
                "reason": "current_science_missing"
                if s["state"] == "missing"
                else "raw_or_custody_disqualified"
                if s["state"] == "disqualified"
                else "cold_recomputed",
            }
            for s in sources
        ],
        "mutation_results": private_mutations(),
        "acceptance_gate_results": {
            "validity": not any(s["state"] == "disqualified" for s in sources),
            "readiness": ready,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "independent_evidence_ready_score": ready,
        "sample_size_budget": {
            "intended": {"Exp7799": 64, "Exp7801": 48, "Exp7802": 64},
            "eligible": sum(
                part["independent_n"]
                for sections in branches.values()
                for part in sections.values()
            ),
            "started": 0,
            "completed": 0,
            "excluded": 0,
            "censored": 0,
            "independent_n": 0,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "root": str(root.resolve()),
            "backend": "host aggregation",
            "declared_paths": list(PLAN.values()),
            "resource_check": "CPU and local file access",
            "source_count": len(sources),
        },
        "claim_scope": {
            "natural_annotations": "exposed_development_only",
            "all_640_source_families_exposed": True,
            "fresh_generalization_eligible": False,
            "fixtures": "circular_positive",
        },
        "verifier_is_oracle": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            key: 0
            for key in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "random_seed": {"bootstrap": 67807, "permutation": 67807},
        "duration_s": 0.0,
        "phase_spans": [],
        "validation_receipts": {
            "frozen_affected_scope": {},
            "required_commands": [],
            "full_python_suite": [],
            "terminal_readers": [],
            "e2e_checks": [],
        },
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "code": sha256_file(Path(__file__)),
            "sources": sources,
            "roles": ["evaluation64", "retention32", "qwen48"],
            "configuration": {"cost": [5, 1, 0.25], "seed": 67807},
        }
    )
    artifact["field_principles"] = {
        key: "Bind this field to exact current evidence and its measured scope." for key in artifact
    }
    artifact["gate_principles"] = {
        key: "A fixture or missing producer cannot prove natural benefit." for key in GATES
    }
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """Reopen current source bytes and compare stable claims in a fresh process."""
    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    sources, failures = inspect_sources(root)
    branches, raw_failures = read_branches(root, sources)
    expected = build_artifact(root, value["run_date"], sources, failures + raw_failures, branches)
    keys = (
        "source_artifact_hashes",
        "gate_check_summary",
        "rows",
        "discrepancy_rows",
        "branch_dispositions",
        "mutation_results",
        "sample_size_budget",
        "independent_evidence_ready_score",
        "reproducibility_checksum",
    )
    return [f"{key}_changed" for key in keys if value.get(key) != expected[key]]
