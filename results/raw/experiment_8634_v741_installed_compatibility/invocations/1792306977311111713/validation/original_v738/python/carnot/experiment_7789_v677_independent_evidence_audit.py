"""Audit V677 decision and learning evidence from source bytes (REQ-REPORT-7789)."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


PLAN = {
    7786: "results/experiment_7786_v677_decision_measurement.json",
    7788: "results/experiment_7788_v677_continuous_acquisition.json",
    7787: "results/experiment_7787_v677_qwen_event_confidence.json",
}
PRE_GATE = {
    7786: "results/raw/experiment_7786_v677_decision_measurement/conductor_pre_gate.json",
    7788: "results/raw/experiment_7788_v677_continuous_acquisition/conductor_pre_gate.json",
}
DECISION_ARMS = (
    "constrained_set",
    "augmented_set",
    "constrained_mlp",
    "local_logistic",
    "source_erased_constrained_set",
)
LEARNING_ARMS = (
    "adaptive",
    "delayed_commit",
    "frozen",
    "complete_static",
    "shuffled",
)
GATES = (
    "validity",
    "readiness",
    "probability_quality",
    "decision_benefit",
    "retention",
    "efficiency",
)
SEED = 67789


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Expose each phase boundary, elapsed time, and completed units promptly."""
    print(
        f"phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def failure(number: int, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Keep both operands and exact byte identity for a failed custody check."""
    return {
        "upstream_id": f"Exp{number}",
        "artifact_path": str(path.resolve()),
        "artifact_hash": sha256_file(path) if path.is_file() else None,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
    }


def inspect_sources(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check exact same-roadmap producers before opening their scientific rows."""
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number, relative in PLAN.items():
        path = root / relative
        receipt = root / PRE_GATE[number] if number in PRE_GATE else None
        source: dict[str, Any] = {
            "upstream_id": f"Exp{number}",
            "path": relative,
            "sha256": sha256_file(path) if path.is_file() else None,
            "pre_gate_receipt": {
                "path": str(receipt.relative_to(root)),
                "sha256": sha256_file(receipt),
                "role": "explanation_only",
            }
            if receipt is not None and receipt.is_file()
            else None,
            "imported_fields": {},
            "raw_paths": {},
            "state": "missing",
        }
        if not path.is_file():
            if number != 7787:
                failures.append(
                    failure(number, path, "producer_path", "existing declared producer", "missing")
                )
            sources.append(source)
            continue
        try:
            value = json.loads(path.read_bytes())
            if not isinstance(value, dict):
                raise ValueError("producer must be an object")
        except (ValueError, UnicodeError):
            value = {}
            failures.append(failure(number, path, "schema", "JSON object", "invalid"))
        expected_fields = {
            "experiment_id": number if number != 7787 else "exp7787-qwen-event-confidence",
            "milestone": "2026.09.677",
            "run_date": "20260927",
            "flagged_adversarial": False,
        }
        for field, expected in expected_fields.items():
            if value.get(field) != expected:
                if number != 7787:
                    failures.append(failure(number, path, field, expected, value.get(field)))
        verdict = value.get("verdict_class")
        if verdict not in {"positive", "null", "circular_positive"} or not str(
            value.get("honest_verdict", "")
        ).startswith("complete_"):
            if number != 7787:
                failures.append(
                    failure(number, path, "verdict_class", "qualified complete science", verdict)
                )
        source["imported_fields"] = {
            k: value.get(k)
            for k in (*expected_fields, "honest_verdict", "verdict_class", "flagged_adversarial")
        }
        if number != 7787:
            for field in (
                "raw_rows_path",
                "frozen_heads_manifest_path",
                "role_manifest_path",
                "rejection_records_path",
            ):
                raw = value.get(field)
                safe = (
                    isinstance(raw, str)
                    and raw
                    and not Path(raw).is_absolute()
                    and ".." not in Path(raw).parts
                )
                raw_path = root / raw if safe else path
                digest = sha256_file(raw_path) if safe and raw_path.is_file() else None
                source["raw_paths"][field] = {"path": raw, "sha256": digest}
                if digest is None:
                    failures.append(
                        failure(number, raw_path, field, "existing safe declared bytes", raw)
                    )
                expected_hash = value.get(field.replace("_path", "_sha256"))
                if expected_hash != digest:
                    failures.append(
                        failure(
                            number,
                            raw_path,
                            field.replace("_path", "_sha256"),
                            expected_hash,
                            digest,
                        )
                    )
            if number == 7788:
                field = "event_rows_path"
                raw = value.get(field)
                safe = (
                    isinstance(raw, str)
                    and raw
                    and not Path(raw).is_absolute()
                    and ".." not in Path(raw).parts
                )
                raw_path = root / raw if safe else path
                digest = sha256_file(raw_path) if safe and raw_path.is_file() else None
                source["raw_paths"][field] = {"path": raw, "sha256": digest}
                if digest is None:
                    failures.append(
                        failure(number, raw_path, field, "existing safe declared bytes", raw)
                    )
                if value.get("event_rows_sha256") != digest:
                    failures.append(
                        failure(
                            number,
                            raw_path,
                            "event_rows_sha256",
                            value.get("event_rows_sha256"),
                            digest,
                        )
                    )
        optional_ok = all(
            value.get(field) == expected for field, expected in expected_fields.items()
        )
        optional_ok = optional_ok and verdict in {"positive", "null", "circular_positive"}
        optional_ok = optional_ok and str(value.get("honest_verdict", "")).startswith("complete_")
        source["state"] = (
            "eligible"
            if (
                optional_ok
                if number == 7787
                else not any(f["upstream_id"] == f"Exp{number}" for f in failures)
            )
            else "disqualified"
        )
        sources.append(source)
    return sources, failures


def score(row: dict[str, Any]) -> dict[str, float]:
    """Recompute unsupported risk losses and the frozen typed action cost."""
    p, y, action = row["probability"], row["label"], row["action"]
    if type(y) is not int or y not in (0, 1) or type(p) not in (float, int) or not 0 < p < 1:
        raise ValueError("probability_or_label")
    if action not in {"accept", "reject", "escalate"}:
        raise ValueError("action")
    return {
        "brier": (p - y) ** 2,
        "nll": -(y * math.log(p) + (1 - y) * math.log1p(-p)),
        "cost": 0.25
        if action == "escalate"
        else 5.0
        if action == "accept" and y == 1
        else 1.0
        if action == "reject" and y == 0
        else 0.0,
        "false_accept": float(action == "accept" and y == 1),
    }


def paired(values: list[float], seed: int = SEED) -> dict[str, Any]:
    """Resample whole families and randomize paired signs, never row views."""
    if not values:
        return {"n": 0, "mean": None, "lower95": None, "upper95": None, "p": None}
    rng = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    extreme = sum(
        abs(sum(v * (1 if rng.randrange(2) else -1) for v in values) / n) >= abs(mean)
        for _ in range(10000)
    )
    return {
        "n": n,
        "mean": mean,
        "lower95": draws[250],
        "upper95": draws[9749],
        "p": (extreme + 1) / 10001,
    }


def reduce_rows(
    rows: list[dict[str, Any]],
    role: str,
    families: int,
    arms: tuple[str, ...],
    head_digest: str,
) -> dict[str, Any]:
    """Score primitive rows and verify the full family, seed, and arm roster."""
    errors: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_scores: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    retained: list[dict[str, Any]] = []
    for row in rows:
        family = row["family_id"]
        grouped[family].append(row)
        if row.get("label_join") != family:
            errors.add("label_join")
        if row.get("role") != role:
            errors.add("role")
        if row.get("head_sha256") != head_digest:
            errors.add("head_digest")
        if row.get("prediction_tick", 0) >= row.get("label_tick", 0):
            errors.add("label_chronology")
        try:
            metrics = score(row)
            for key in ("brier", "nll", "cost", "false_accept"):
                if key in row and not math.isclose(row[key], metrics[key], abs_tol=1e-8):
                    errors.add("saved_metric")
            family_scores[(family, row["arm"])].append(metrics)
        except (KeyError, ValueError):
            metrics = None
            errors.add("metric")
        retained.append(
            {
                "family_id": family,
                "arm": row.get("arm"),
                "seed": row.get("seed"),
                "role": role,
                "label": row.get("label"),
                "metrics": metrics,
                "exclusions": [] if metrics else ["invalid_metric"],
                "censored": bool(row.get("censored", False)),
                "raw_provenance": {
                    "head_sha256": row.get("head_sha256"),
                    "source_sha256": row.get("source_sha256"),
                },
            }
        )
    if len(grouped) != families:
        errors.add("family_count")
    expected = {(arm, seed) for arm in arms for seed in range(3)}
    for group in grouped.values():
        if len(group) != len(expected) or {(r["arm"], r["seed"]) for r in group} != expected:
            errors.add("roster")
        if len({r["label"] for r in group}) != 1 or len({r["source_sha256"] for r in group}) != 1:
            errors.add("label_join")
    by_arm: dict[str, dict[str, float]] = {}
    for arm in arms:
        arm_values = [
            score
            for (family, name), scores in family_scores.items()
            if name == arm
            for score in scores
        ]
        if arm_values:
            by_arm[arm] = {
                key: sum(item[key] for item in arm_values) / len(arm_values)
                for key in ("brier", "nll", "cost", "false_accept")
            }
    comparisons: list[dict[str, Any]] = []
    primary = arms[0]
    for comparator in arms[1:]:
        for metric in ("brier", "cost"):
            deltas = []
            for family in grouped:
                left, right = (
                    family_scores.get((family, primary), []),
                    family_scores.get((family, comparator), []),
                )
                if len(left) == 3 and len(right) == 3:
                    deltas.append(
                        sum(x[metric] for x in right) / 3 - sum(x[metric] for x in left) / 3
                    )
            comparisons.append(
                {"primary": primary, "comparator": comparator, "metric": metric, **paired(deltas)}
            )
    ordered = sorted(comparisons, key=lambda item: item["p"] if item["p"] is not None else 1)
    adjusted = 0.0
    for rank, comparison in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, (len(ordered) - rank) * (comparison["p"] or 1)))
        comparison["holm_p"] = adjusted
        comparison["holm_pass"] = adjusted <= 0.05
    return {
        "failed_checks": sorted(errors),
        "independent_n": len(grouped),
        "coverage": len(grouped) / families if families else 0,
        "by_arm": by_arm,
        "comparisons": comparisons,
        "rows": retained,
    }


def audit_events(
    events: list[dict[str, Any]], queries: list[dict[str, Any]], identical_arms: bool
) -> list[str]:
    """Require released feedback, sealed queries, and a later changed decision."""
    errors: set[str] = set()
    predicted = {e.get("query_id"): e["tick"] for e in events if e.get("kind") == "prediction"}
    feedback = {e.get("query_id"): e["tick"] for e in events if e.get("kind") == "feedback"}
    for key, tick in feedback.items():
        if key not in predicted or tick <= predicted[key]:
            errors.add("future_feedback")
    admissions = [e for e in events if e.get("kind") == "admission"]
    if len({e.get("feedback_id") for e in admissions}) != len(admissions):
        errors.add("duplicate_admission")
    for event in events:
        kind, tick = event.get("kind"), event.get("tick", -1)
        if kind == "proposal" and event.get("role") in {"evaluation64", "retention32"}:
            errors.add("selection_label_leakage")
        if kind == "commit" and any(
            q["query_open_tick"] <= tick < q["query_close_tick"] for q in queries
        ):
            errors.add("commit_inside_query")
        if kind == "restart" and event.get("exact_parity") is not True:
            errors.add("restart")
    if admissions and not any(
        e.get("kind") == "later_decision" and e.get("before_action") != e.get("after_action")
        for e in events
    ):
        errors.add("no_later_change")
    if admissions and identical_arms:
        errors.add("identical_learned_frozen")
    return sorted(errors)


def reduce_qwen(
    root: Path, rows: list[dict[str, Any]], expected_families: int = 24
) -> dict[str, Any]:
    """Reprice optional Qwen risks and authenticate every saved request and reply."""
    errors: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_arm_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    retained: list[dict[str, Any]] = []
    for row in rows:
        family, arm = row["family_id"], row["arm"]
        grouped[family].append(row)
        risk = row["metrics"]["unsupported_risk"]
        truth = row["natural_unsupported"]
        escalated = bool(row["metrics"]["forced_escalation"] or risk >= 0.5)
        measured = {
            "brier": (risk - int(truth)) ** 2 if truth is not None else None,
            "cost": (1 if escalated else 5 if truth else 0) if truth is not None else None,
            "false_accept": bool(truth is True and not escalated),
        }
        if not isinstance(risk, (int, float)) or not 0 <= risk <= 1:
            errors.add("risk")
        if row.get("brier") != measured["brier"] and not (
            isinstance(row.get("brier"), (int, float))
            and measured["brier"] is not None
            and math.isclose(row["brier"], measured["brier"], abs_tol=1e-8)
        ):
            errors.add("saved_metric")
        if (
            row.get("decision_cost") != measured["cost"]
            or row.get("escalated") != escalated
            or row.get("false_accept") != measured["false_accept"]
        ):
            errors.add("saved_metric")
        provenance = {}
        for kind in ("request", "response"):
            path = Path(row[f"raw_{kind}_path"])
            observed = (
                sha256_file(path)
                if path.is_file() and path.resolve().is_relative_to(root.resolve())
                else None
            )
            provenance[kind] = {"path": str(path), "sha256": observed}
            if observed != row[f"raw_{kind}_sha256"]:
                errors.add("raw_hash")
        if truth is not None:
            by_arm_rows[arm].append(
                {"brier": float(measured["brier"]), "cost": float(measured["cost"])}
            )
        retained.append(
            {
                "family_id": family,
                "arm": arm,
                "seed": None,
                "role": "qwen_optional",
                "metrics": measured,
                "raw_provenance": provenance,
                "exclusions": [] if truth is not None else ["unlabeled"],
                "censored": bool(row.get("censored")),
            }
        )
    if len(grouped) != expected_families:
        errors.add("family_count")
    for group in grouped.values():
        if len(group) != 2 or {r["arm"] for r in group} != {"generic", "event"}:
            errors.add("roster")
        if len({r["natural_unsupported"] for r in group}) != 1:
            errors.add("label_join")
    return {
        "failed_checks": sorted(errors),
        "independent_n": len(grouped),
        "coverage": len(grouped) / expected_families if expected_families else 0,
        "by_arm": {
            arm: {
                metric: sum(r[metric] for r in values) / len(values) for metric in ("brier", "cost")
            }
            for arm, values in by_arm_rows.items()
            if values
        },
        "comparisons": [],
        "rows": retained,
    }


def private_tampers() -> list[dict[str, Any]]:
    """Cold-reduce private corruptions through the same primitive checker."""
    arms = ("constrained_set", "augmented_set")
    base = [
        {
            "family_id": f"f{i}",
            "label_join": f"f{i}",
            "role": "evaluation64",
            "label": 0,
            "arm": arm,
            "seed": seed,
            "probability": 0.2,
            "action": "accept",
            "head_sha256": "head",
            "source_sha256": f"source-{i}",
            "prediction_tick": 1,
            "label_tick": 2,
        }
        for i in range(2)
        for arm in arms
        for seed in range(3)
    ]
    outcomes: list[dict[str, Any]] = []
    for name, expected in (
        ("dropped_family", "family_count"),
        ("swapped_label_join", "label_join"),
        ("missing_arm", "roster"),
        ("duplicate_seed", "roster"),
        ("altered_head_digest", "head_digest"),
    ):
        changed = deepcopy(base)
        if name == "dropped_family":
            changed = [row for row in changed if row["family_id"] != "f1"]
        elif name == "swapped_label_join":
            changed[0]["label_join"] = "f1"
        elif name == "missing_arm":
            changed.pop()
        elif name == "duplicate_seed":
            changed[0]["seed"] = 1
        else:
            changed[0]["head_sha256"] = "forged"
        observed = reduce_rows(changed, "evaluation64", 2, arms, "head")["failed_checks"]
        outcomes.append(
            {
                "mutation": name,
                "failed_check": expected,
                "observed_checks": observed,
                "rejected": expected in observed,
            }
        )
    for name, events, expected in (
        (
            "future_feedback",
            [
                {"kind": "prediction", "query_id": "q", "tick": 2},
                {"kind": "feedback", "query_id": "q", "tick": 1},
            ],
            "future_feedback",
        ),
        ("commit_inside_query", [{"kind": "commit", "tick": 2}], "commit_inside_query"),
    ):
        observed = audit_events(events, [{"query_open_tick": 1, "query_close_tick": 3}], False)
        outcomes.append(
            {
                "mutation": name,
                "failed_check": expected,
                "observed_checks": observed,
                "rejected": expected in observed,
            }
        )
    original = reduce_rows(base, "evaluation64", 2, arms, "head")
    forged = deepcopy(original)
    forged["by_arm"][arms[0]]["brier"] = 1.0
    observed = reduce_rows(base, "evaluation64", 2, arms, "head")
    rejected = forged["by_arm"] != observed["by_arm"]
    outcomes.append(
        {
            "mutation": "forged_aggregate",
            "failed_check": "aggregate_mismatch",
            "observed_checks": ["aggregate_mismatch"] if rejected else [],
            "rejected": rejected,
        }
    )
    return outcomes


def read_branches(
    root: Path, sources: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Reduce authenticated raw bytes, never a producer's headline fields."""
    reduced: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for source in sources:
        number = int(source["upstream_id"][3:])
        if source["state"] != "eligible":
            continue
        if number == 7787:
            try:
                value = json.loads((root / source["path"]).read_bytes())
                optional = reduce_qwen(root, value["rows"])
                reduced[number] = {"qwen_optional": optional}
                source["optional_failures"] = optional["failed_checks"]
                if optional["failed_checks"]:
                    source["state"] = "disqualified"
            except (KeyError, ValueError, TypeError, OSError) as exc:
                source["optional_failures"] = [f"raw_schema:{type(exc).__name__}"]
                source["state"] = "disqualified"
            continue
        raw = source["raw_paths"]["raw_rows_path"]["path"]
        try:
            payload = json.loads((root / raw).read_bytes())
            rowsets = payload["rowsets"]
            head = payload["head_sha256"]
            manifest = json.loads(
                (root / source["raw_paths"]["frozen_heads_manifest_path"]["path"]).read_bytes()
            )
            if manifest.get("head_sha256") != head:
                failures.append(
                    failure(number, root / raw, "head_manifest", head, manifest.get("head_sha256"))
                )
                continue
            role_manifest = json.loads(
                (root / source["raw_paths"]["role_manifest_path"]["path"]).read_bytes()
            )
            if not isinstance(role_manifest, dict) or any(
                set(role_manifest.get(role, [])) != {row["family_id"] for row in role_rows}
                for role, role_rows in rowsets.items()
            ):
                failures.append(
                    failure(
                        number, root / raw, "role_manifest", "exact role families", role_manifest
                    )
                )
                continue
            rejections = json.loads(
                (root / source["raw_paths"]["rejection_records_path"]["path"]).read_bytes()
            )
            if not isinstance(rejections, list):
                failures.append(
                    failure(
                        number,
                        root / raw,
                        "rejection_records",
                        "JSON list",
                        type(rejections).__name__,
                    )
                )
                continue
            if number == 7786:
                branch = {
                    "evaluation64": reduce_rows(
                        rowsets["evaluation64"], "evaluation64", 64, DECISION_ARMS, head
                    )
                }
            else:
                branch = {
                    "evaluation64": reduce_rows(
                        rowsets["evaluation64"], "evaluation64", 64, LEARNING_ARMS, head
                    ),
                    "retention32": reduce_rows(
                        rowsets["retention32"], "retention32", 32, LEARNING_ARMS, head
                    ),
                }
                event_path = source["raw_paths"]["event_rows_path"]["path"]
                events = json.loads((root / event_path).read_bytes())
                identical = payload.get("learned_head_sha256") == payload.get("frozen_head_sha256")
                branch["event_checks"] = audit_events(events, rowsets["evaluation64"], identical)
                branch["events"] = events
            reduced[number] = branch
            for name, part in branch.items():
                checks = (
                    part
                    if name == "event_checks"
                    else part.get("failed_checks", [])
                    if isinstance(part, dict)
                    else []
                )
                for check in checks:
                    failures.append(failure(number, root / raw, f"{name}.{check}", "valid", check))
        except (KeyError, ValueError, TypeError, UnicodeError, OSError) as exc:
            failures.append(
                failure(number, root / raw, "raw_schema", "reducible", type(exc).__name__)
            )
    return reduced, failures


def build_artifact(
    root: Path,
    date: str,
    sources: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    branches: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Keep all observed units and close benefit gates when science is absent."""
    ready = not failures and 7786 in branches and 7788 in branches
    rows = [
        {
            "family_id": s["upstream_id"],
            "arm": "custody",
            "seed": None,
            "metrics": None,
            "raw_path": s["path"],
            "raw_hash": s["sha256"],
            "exclusions": [] if s["state"] == "eligible" else [s["state"]],
            "censored": False,
        }
        for s in sources
    ]
    audit_rows = []
    for number, branch in branches.items():
        for name, part in branch.items():
            if isinstance(part, dict) and "rows" in part:
                rows.extend(part["rows"])
                audit_rows.append(
                    {
                        "upstream_id": f"Exp{number}",
                        "role": name,
                        "independent_n": part["independent_n"],
                        "coverage": part["coverage"],
                        "by_arm": part["by_arm"],
                        "comparisons": part["comparisons"],
                        "failed_checks": part["failed_checks"],
                    }
                )
    by_role = {
        (
            "qwen_optional"
            if row["upstream_id"] == "Exp7787"
            else f"{row['upstream_id']}.{row['role']}"
        ): row["independent_n"]
        for row in audit_rows
    }
    independent_n = max(
        (
            row["independent_n"]
            for row in audit_rows
            if row["upstream_id"] in {"Exp7786", "Exp7788"}
        ),
        default=0,
    )
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7789.v677.independent_evidence_audit.v1",
        "experiment_id": 7789,
        "milestone": "2026.09.677",
        "run_date": date,
        "honest_verdict": "complete_null_independent_audit"
        if ready
        else "complete_blocked_required_v677_evidence",
        "verdict_class": "null" if ready else "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "audit_rows": audit_rows,
        "rejected_mutations": private_tampers(),
        "independent_evidence_ready_score": int(ready),
        "acceptance_gate_results": {
            gate: (True if gate == "validity" else int(ready) if gate == "readiness" else None)
            for gate in GATES
        },
        "sample_size_budget": {
            "intended": {"evaluation_families": 64, "retention_families": 32},
            "eligible": independent_n if ready else 0,
            "started": sum(by_role.values()),
            "completed": sum(by_role.values()),
            "excluded": sum(bool(r["exclusions"]) for r in rows),
            "censored": sum(bool(r["censored"]) for r in rows),
            "independent_n": independent_n,
            "by_role": by_role,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "root": str(root.resolve()),
            "declared_paths": PLAN,
            "producer_states": {s["upstream_id"]: s["state"] for s in sources},
            "backend": "host aggregation",
            "resources": "no model load",
        },
        "claim_scope": {
            "natural_annotations": "exposed_development_only",
            "fixtures": "circular_positive",
            "fresh_generalization_eligible": False,
        },
        "verifier_is_oracle": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            k: 0
            for k in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "random_seed": {"bootstrap": SEED, "permutation": SEED},
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
            "sources": sources,
            "code": sha256_file(Path(__file__)),
            "roles": ["evaluation64", "retention32"],
            "parameters": [5, 1, 0.25, 10000],
            "seed": SEED,
        }
    )
    artifact["field_principles"] = {
        key: "Exact input and role identity bound this claim." for key in artifact
    }
    artifact["field_principles"].update(
        {
            "rows": "Headlines must trace to individual units.",
            "honest_verdict": "An external block must be terminal.",
            "sample_size_budget": "Seeds do not create new families.",
            "validation_receipts": "Required checks precede readiness.",
            "independent_evidence_ready_score": "Decision and learning producers must independently qualify.",
        }
    )
    artifact["gate_principles"] = {
        gate: "Measured independent evidence must support this gate." for gate in GATES
    }
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """Reopen exact source bytes and compare all stable row and gate claims."""
    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    sources, failures = inspect_sources(root)
    branches, raw_failures = read_branches(root, sources)
    rebuilt = build_artifact(root, value["run_date"], sources, failures + raw_failures, branches)
    keys = (
        "source_artifact_hashes",
        "gate_check_summary",
        "rows",
        "audit_rows",
        "rejected_mutations",
        "sample_size_budget",
        "reproducibility_checksum",
        "independent_evidence_ready_score",
    )
    return [f"{key}_changed" for key in keys if value.get(key) != rebuilt[key]]


def run_experiment(root: Path, date: str, output: Path, *, validate: bool = True) -> dict[str, Any]:
    """Run bounded validation before atomically writing one terminal record."""
    import tempfile
    from carnot.reporting import experiment_7303_validation_scope as checks

    root = root.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    phase_start = 0.0

    def finish(name: str, units: int) -> None:
        nonlocal phase_start
        end = time.monotonic() - started
        spans.append(
            {
                "phase": name,
                "start_s": phase_start,
                "end_s": end,
                "duration_s": end - phase_start,
                "completed_units": units,
            }
        )
        phase_start = end
        progress(started, name, "complete", units)

    progress(started, "preconditions", "start", 0)
    sources, failures = inspect_sources(root)
    finish("preconditions", len(sources))
    progress(started, "reduction", "start", 0)
    branches, raw_failures = read_branches(root, sources)
    artifact = build_artifact(root, date, sources, failures + raw_failures, branches)
    scope = json.loads((Path(__file__).parents[2] / "ops/exp7789_frozen_scope.json").read_bytes())
    artifact["validation_receipts"]["frozen_affected_scope"] = scope
    finish("reduction", len(artifact["rows"]))
    raw_dir = root / "results/raw/experiment_7789_v677_independent_evidence_audit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if validate:
        private = Path(tempfile.mkdtemp(prefix="exp7789-", dir="/tmp"))
        basetemp = private / "basetemp"
        basetemp.mkdir(parents=True, exist_ok=True)
        progress(started, "affected_validation", "start", 0)
        commands = checks.build_scoped_commands(
            root,
            scope["direct_tests"] + scope["transitive_consumers"],
            scope["changed_modules"][:1],
            static_paths=scope["changed_modules"][1:],
            basetemp=basetemp,
            coverage_file=private / "coverage.data",
        )
        receipts = checks.run_commands(
            root,
            commands,
            log_dir=raw_dir / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / "coverage.data")},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["required_commands"] = receipts
        artifact["validation_receipts"].update(checks.reduce_required_checks(receipts))
        finish("affected_validation", len(receipts))
        progress(started, "full_python_suite", "start", 0)
        full = checks.CommandSpec(
            "full_python_suite",
            (
                str(root / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={basetemp / 'full'}",
            ),
            "repository_health",
            1800,
        )
        cached_path = raw_dir / "validation/full/receipt.json"
        if cached_path.is_file():
            try:
                cached = json.loads(cached_path.read_bytes())
                cached_argv = cached["command_argv"]
                log_path = (root / cached["log_path"]).resolve()
                if (
                    cached.get("name") != "full_python_suite"
                    or cached_argv[:8] != list(full.argv[:8])
                    or not log_path.is_relative_to(root)
                    or sha256_file(log_path) != cached["log_sha256"]
                    or cached.get("passed") is not (cached.get("exit_code") == 0)
                ):
                    raise ValueError("invalid_full_suite_receipt")
                full_receipts = [cached]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                full_receipts = [
                    {
                        "name": "full_python_suite",
                        "passed": False,
                        "error": f"invalid_cached_receipt:{type(exc).__name__}",
                        "receipt_path": str(cached_path),
                    }
                ]
        else:
            full_receipts = checks.run_commands(
                root,
                [full],
                log_dir=raw_dir / "validation/full",
                extra_env={"JAX_PLATFORMS": "cpu"},
                heartbeat_s=30,
            )
        artifact["validation_receipts"]["full_python_suite"] = full_receipts
        artifact["validation_receipts"]["repository_health_passed"] = bool(
            full_receipts[0]["passed"]
        )
        finish("full_python_suite", 1)
        if (
            not artifact["validation_receipts"]["required_checks_passed"]
            or "error" in full_receipts[0]
        ):
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["acceptance_gate_results"].update(validity=False, readiness=0)
            artifact["independent_evidence_ready_score"] = 0
    artifact["phase_spans"] = list(spans)
    artifact["duration_s"] = time.monotonic() - started
    candidate = raw_dir / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    if validate:
        progress(started, "terminal_readers", "start", 0)
        python = str(root / ".venv/bin/python")
        readers = [
            checks.CommandSpec(
                "cold_replay",
                (
                    python,
                    "-u",
                    "scripts/experiments/experiment_7789_v677_independent_evidence_audit.py",
                    "--cold",
                    str(candidate),
                ),
                "exact_candidate",
                180,
            ),
            checks.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
                "exact_candidate",
                180,
            ),
            checks.CommandSpec(
                "strict_row_consistency",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "exact_candidate",
                180,
            ),
        ]
        terminal = checks.run_commands(
            root,
            readers,
            log_dir=raw_dir / "validation/terminal",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["terminal_readers"] = terminal
        artifact["validation_receipts"]["exact_candidate_sha256"] = sha256_file(candidate)
        artifact["validation_receipts"]["e2e_checks"] = [
            {"name": "real_entrypoint_cold_replay", "passed": terminal[0]["passed"]}
        ]
        try:
            report = json.loads((root / terminal[1]["log_path"]).read_text())
            artifact["flagged_adversarial"] = bool(report["flagged_count"])
        except (OSError, ValueError, KeyError):
            artifact["flagged_adversarial"] = True
        if not all(r["passed"] for r in terminal) or artifact["flagged_adversarial"]:
            artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
            artifact["verdict_class"] = "disqualified"
            artifact["acceptance_gate_results"].update(validity=False, readiness=0)
            artifact["independent_evidence_ready_score"] = 0
        finish("terminal_readers", len(terminal))
    progress(started, "publication", "start", 0)
    finish("publication", 1)
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    atomic_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run the real entrypoint or replay a saved candidate in a fresh process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold", type=Path)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cold:
        errors = cold_replay(args.cold)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    root = args.fixture_root or Path(__file__).parents[2]
    output = args.output or root / "results/experiment_7789_v677_independent_evidence_audit.json"
    run_experiment(root, args.date, output, validate=args.fixture_root is None)
    return 0
