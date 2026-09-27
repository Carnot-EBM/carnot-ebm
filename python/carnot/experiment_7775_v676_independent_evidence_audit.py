"""Audit V676 raw decisions and delayed acquisition without producer reducers.

REQ-REPORT-7775. Exact current producers own the science. This module can
explain absent inputs, but it cannot turn a conductor skip into measurements.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

PLAN = {
    7772: "results/experiment_7772_v676_decision_measurement.json",
    7774: "results/experiment_7774_v676_continuous_acquisition.json",
}
PRE_GATE = {number: f"results/experiment_{number}_pre_gate.json" for number in PLAN}
RAW_KEYS = {7772: ("raw_rows_path",), 7774: ("raw_rows_path", "event_rows_path")}
SCOPE_PATH = (
    "results/raw/experiment_7775_v676_independent_evidence_audit/frozen_affected_scope.json"
)
GATES = (
    "validity",
    "readiness",
    "probability_quality",
    "decision_benefit",
    "retention",
    "efficiency",
)


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Show each boundary so an operator can distinguish progress from a stall."""
    print(
        f"[exp7775] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def failure(number: int, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Keep the observed operand and exact bytes behind each custody failure."""
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
    """Check only the two paths in the matching roadmap, never nearby results."""
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number, relative in PLAN.items():
        path = root / relative
        receipt = root / PRE_GATE[number]
        source: dict[str, Any] = {
            "upstream_id": f"Exp{number}",
            "path": relative,
            "sha256": sha256_file(path) if path.is_file() else None,
            "run_date": None,
            "state": "missing",
            "eligible": False,
            "imported_fields": {},
            "raw_paths": {},
            "pre_gate_receipt": {
                "path": PRE_GATE[number],
                "sha256": sha256_file(receipt),
                "role": "explanation_only",
            }
            if receipt.is_file()
            else None,
        }
        if not path.is_file():
            failures.append(
                failure(number, path, "producer_path", "existing declared producer", "missing")
            )
        else:
            try:
                value = json.loads(path.read_bytes())
                if not isinstance(value, dict):
                    raise ValueError("producer must be a JSON object")
            except (ValueError, UnicodeError) as exc:
                value = {}
                failures.append(failure(number, path, "schema", "JSON object", type(exc).__name__))
            source["run_date"] = value.get("run_date")
            source["state"] = "present"
            expected_fields = {
                "experiment_id": number,
                "milestone": "2026.09.676",
                "run_date": "20260927",
                "flagged_adversarial": False,
            }
            for field, expected in expected_fields.items():
                observed = value.get(field)
                source["imported_fields"][field] = observed
                if observed != expected:
                    failures.append(failure(number, path, field, expected, observed))
            for field in ("honest_verdict", "verdict_class", *RAW_KEYS[number]):
                source["imported_fields"][field] = value.get(field)
            verdict = value.get("honest_verdict")
            if not isinstance(verdict, str) or not verdict.startswith("complete_"):
                failures.append(failure(number, path, "honest_verdict", "complete_*", verdict))
            if value.get("verdict_class") not in {"positive", "null", "circular_positive"}:
                failures.append(
                    failure(
                        number,
                        path,
                        "verdict_class",
                        "positive|null|circular_positive",
                        value.get("verdict_class"),
                    )
                )
            for key in RAW_KEYS[number]:
                raw = value.get(key)
                if (
                    not isinstance(raw, str)
                    or not raw
                    or Path(raw).is_absolute()
                    or ".." in Path(raw).parts
                ):
                    failures.append(failure(number, path, key, "safe declared raw path", raw))
                    continue
                raw_path = root / raw
                digest = sha256_file(raw_path) if raw_path.is_file() else None
                source["raw_paths"][key] = {"path": raw, "sha256": digest}
                if digest is None:
                    failures.append(failure(number, raw_path, key, "existing raw bytes", "missing"))
                expected_hash = value.get(key.replace("_path", "_sha256"))
                if expected_hash != digest:
                    failures.append(
                        failure(
                            number, raw_path, key.replace("_path", "_sha256"), expected_hash, digest
                        )
                    )
            source["eligible"] = not any(f["upstream_id"] == f"Exp{number}" for f in failures)
            source["state"] = "eligible" if source["eligible"] else "disqualified"
        sources.append(source)
    return sources, failures


def score(row: dict[str, Any]) -> dict[str, float]:
    """Recompute fixed losses from a binary label, risk, and saved action."""
    p, y, action = row["probability"], row["label"], row["action"]
    if type(y) is not int or y not in (0, 1) or type(p) not in (int, float) or not 0 < p < 1:
        raise ValueError("probability")
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
        "false_accepts": float(action == "accept" and y == 1),
    }


def paired(values: list[float], seed: int = 67675) -> dict[str, Any]:
    """Draw families, not repeated views, for uncertainty and a paired sign test."""
    if not values:
        return {"n": 0, "mean": None, "lower95": None, "upper95": None, "p": None}
    n = len(values)
    rng = random.Random(seed)
    draws = sorted(sum(values[rng.randrange(n)] for _ in values) / n for _ in range(10000))
    signs = [value > 0 for value in values if value != 0]
    k = min(sum(signs), len(signs) - sum(signs))
    p = (
        min(1.0, 2 * sum(math.comb(len(signs), i) for i in range(k + 1)) / 2 ** len(signs))
        if signs
        else 1.0
    )
    return {"n": n, "mean": sum(values) / n, "lower95": draws[250], "upper95": draws[9749], "p": p}


def reduce_static(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    """Score every primitive decision, then average paired seeds by family."""
    errors: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    measures: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        family = row["family_id"]
        groups[family].append(row)
        if row.get("label_join") != family:
            errors.add("label_join")
        if row.get("role") != summary["role"]:
            errors.add("role")
        if row.get("prediction_tick", 0) >= row.get("label_tick", 0):
            errors.add("label_chronology")
        try:
            measured = score(row)
            for name in ("brier", "nll", "cost"):
                if name in row and not math.isclose(row[name], measured[name], abs_tol=1e-8):
                    errors.add("saved_metric")
            measures[(family, row["arm"])].append(measured)
        except (ValueError, KeyError):
            errors.add("probability")
    if len(groups) != summary["families"]:
        errors.add("family_count")
    expected = {(arm, seed) for arm in summary["arms"] for seed in summary["seeds"]}
    for group in groups.values():
        if len(group) != len(expected) or {(r["arm"], r["seed"]) for r in group} != expected:
            errors.add("roster")
        if len({r["label"] for r in group}) != 1:
            errors.add("label_join")
        if len({r["source_sha256"] for r in group}) != 1:
            errors.add("source_identity")
        erased = {r["input_hash"] for r in group if r["arm"] == "source_erased"}
        original = {r["input_hash"] for r in group if r["arm"] == "constrained_set"}
        if erased & original:
            errors.add("erased_boundary")
        for row in group:
            if row["arm"] in {"constant", "constant_risk"} and row["probability"] != 0.5:
                errors.add("constant_risk")
    family_means: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (family, arm), values in measures.items():
        family_means[family][arm] = {
            key: sum(v[key] for v in values) / len(values) for key in values[0]
        }
    by_arm = {
        arm: {
            key: sum(f[arm][key] for f in family_means.values() if arm in f)
            / sum(arm in f for f in family_means.values())
            for key in ("brier", "nll", "cost", "false_accepts")
        }
        for arm in summary["arms"]
        if any(arm in f for f in family_means.values())
    }
    tests: dict[str, dict[str, Any]] = {}
    for comparator in summary["arms"]:
        if comparator == "constrained_set":
            continue
        for metric in ("brier", "cost"):
            differences = [
                f[comparator][metric] - f["constrained_set"][metric]
                for f in family_means.values()
                if comparator in f and "constrained_set" in f
            ]
            tests[f"{comparator}_{metric}"] = paired(differences)
    ordered = sorted(
        tests, key=lambda name: tests[name]["p"] if tests[name]["p"] is not None else 1
    )
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (tests[name]["p"] or 1.0) * (len(ordered) - index)))
        tests[name]["holm_p"] = running
    source_test = tests.get("source_erased_brier", {})
    return {
        "failed_checks": sorted(errors),
        "effective_independent_n": len(groups),
        "by_arm": by_arm,
        "paired_tests": tests,
        "family_means": family_means,
        "source_dependence": {
            "brier_advantage": source_test.get("mean"),
            "established": bool(
                source_test.get("lower95") is not None and source_test["lower95"] > 0
            ),
        },
        "view_divergence_is_source_evidence": False,
        "rows": rows,
    }


def reduce_online(
    rows: list[dict[str, Any]], events: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    """Replay admissions only after feedback and demand a later action change."""
    errors: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_arm: dict[str, list[dict[str, float]]] = defaultdict(list)
    family_scores: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    event_keys = {(e.get("kind"), e.get("arm"), e.get("family_id"), e.get("tick")) for e in events}
    feedback = {
        (e.get("family_id"), e.get("arm")): e["tick"] for e in events if e.get("kind") == "feedback"
    }
    for row in rows:
        groups[row["family_id"]].append(row)
        if row.get("label_join") != row["family_id"]:
            errors.add("label_join")
        if row.get("role") != "evaluation64":
            errors.add("role")
        if row["prediction_tick"] >= row["feedback_tick"]:
            errors.add("feedback_chronology")
        for kind, tick in (
            ("prediction", row["prediction_tick"]),
            ("feedback", row["feedback_tick"]),
        ):
            if (kind, row["arm"], row["family_id"], tick) not in event_keys:
                errors.add("event_join")
        try:
            measured = score(row)
            by_arm[row["arm"]].append(measured)
            family_scores[(row["family_id"], row["arm"])].append(measured)
        except (ValueError, KeyError):
            errors.add("probability")
    if len(groups) != summary["families"]:
        errors.add("family_count")
    expected = {(arm, seed) for arm in summary["arms"] for seed in summary["seeds"]}
    for group in groups.values():
        if len(group) != len(expected) or {(r["arm"], r["seed"]) for r in group} != expected:
            errors.add("roster")
    proposals = [e for e in events if e.get("kind") == "proposal"]
    admissions = [e for e in events if e.get("kind") == "admission"]
    if len(proposals) > 8 or len({e["block"] for e in proposals}) != len(proposals):
        errors.add("proposal_credits")
    if len({e["feedback_id"] for e in admissions}) != len(admissions):
        errors.add("one_use_admission")
    if any(
        e["predicate"] not in {p["predicate"] for p in proposals if p["block"] == e["block"]}
        or e["tick"]
        <= min(
            (tick for (family, _arm), tick in feedback.items() if family == e["feedback_id"]),
            default=float("inf"),
        )
        for e in admissions
    ):
        errors.add("queue_order")
    if summary.get("pending_high_water", 0) > 20:
        errors.add("queue_capacity")
    later = [e for e in events if e.get("kind") == "later_prediction"]
    if any(
        not any(
            e["feedback_id"] == a["feedback_id"]
            and e["predicate"] == a["predicate"]
            and e["tick"] > a["tick"]
            and e["before_action"] != e["after_action"]
            for e in later
        )
        for a in admissions
    ):
        errors.add("causal_change")
    if set(summary["static_predicates"]) != set(
        summary.get("complete_static_predicates", summary["static_predicates"])
    ):
        errors.add("static_closure")
    restarts = [e for e in events if e.get("kind") == "restart"]
    if len(restarts) != 1 or restarts[0].get("exact_parity") is not True:
        errors.add("cold_restart")
    means = {
        arm: {key: sum(v[key] for v in values) / len(values) for key in values[0]}
        for arm, values in by_arm.items()
        if values
    }
    family_means: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (family, arm), values in family_scores.items():
        family_means[family][arm] = {
            key: sum(value[key] for value in values) / len(values) for key in values[0]
        }
    tests = {
        f"{comparator}_{metric}": paired(
            [
                family[comparator][metric] - family["adaptive"][metric]
                for family in family_means.values()
                if comparator in family and "adaptive" in family
            ]
        )
        for comparator in ("frozen", "complete_static")
        for metric in ("brier", "cost")
        if comparator in summary["arms"] and "adaptive" in summary["arms"]
    }
    retention = [score(row) for row in summary["retention_rows"]]
    return {
        "failed_checks": sorted(errors),
        "effective_independent_n": len(groups),
        "by_arm": means,
        "family_means": family_means,
        "paired_tests": tests,
        "retention_count": len(retention),
        "retention": {
            key: sum(row[key] for row in retention) / len(retention)
            for key in ("brier", "cost", "false_accepts")
        }
        if retention
        else None,
        "causal_changes": len(admissions) if "causal_change" not in errors else 0,
        "events": events,
        "rows": rows,
    }


def read_branches(
    root: Path, sources: list[dict[str, Any]]
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Open only raw paths authenticated in the matching producer."""
    reductions: dict[int, Any] = {7772: None, 7774: None}
    failures: list[dict[str, Any]] = []
    for number, source in zip(PLAN, sources, strict=True):
        if not source["eligible"]:
            continue
        raw = root / source["raw_paths"]["raw_rows_path"]["path"]
        try:
            value = json.loads(raw.read_bytes())
            if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
                raise ValueError("raw rows object required")
            if number == 7772:
                reduced = reduce_static(value["rows"], value["summary"])
            else:
                events = json.loads(
                    (root / source["raw_paths"]["event_rows_path"]["path"]).read_bytes()
                )
                reduced = reduce_online(value["rows"], events, value["summary"])
            reductions[number] = reduced
            for check in reduced["failed_checks"]:
                failures.append(failure(number, raw, "raw_rows", "valid", check))
        except (ValueError, KeyError, TypeError, UnicodeError, IndexError) as exc:
            failures.append(failure(number, raw, "raw_schema", "reducible", type(exc).__name__))
    return reductions, failures


def build_artifact(
    root: Path,
    date: str,
    sources: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    static: dict[str, Any] | None,
    online: dict[str, Any] | None,
) -> dict[str, Any]:
    """Preserve blocked units so absence never looks like measured benefit."""
    blocked = bool(failures or static is None or online is None)
    static_ok = (
        static is not None
        and not static["failed_checks"]
        and not any(f["upstream_id"] == "Exp7772" for f in failures)
    )
    online_ok = (
        online is not None
        and not online["failed_checks"]
        and online["causal_changes"] > 0
        and online["retention_count"] > 0
        and not any(f["upstream_id"] == "Exp7774" for f in failures)
    )
    rows: list[dict[str, Any]] = [
        {
            "family_id": source["upstream_id"],
            "arm": "custody",
            "seed": None,
            "raw_path": source["path"],
            "raw_hash": source["sha256"],
            "metrics": None,
            "label": None,
            "exclusions": [] if source["eligible"] else [source["state"]],
            "censored": False,
        }
        for source in sources
    ]
    for number, branch in ((7772, static), (7774, online)):
        if branch is None:
            continue
        source = next(s for s in sources if s["upstream_id"] == f"Exp{number}")
        raw = source["raw_paths"]["raw_rows_path"]
        for row in branch["rows"]:
            try:
                metrics = score(row)
            except (ValueError, KeyError):
                metrics = None
            rows.append(
                {
                    "family_id": row["family_id"],
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "raw_path": raw["path"],
                    "raw_hash": raw["sha256"],
                    "metrics": metrics,
                    "label": row["label"],
                    "exclusions": [] if metrics else ["invalid_metric"],
                    "censored": row.get("censored", False),
                }
            )
    independent_n = max(
        static["effective_independent_n"] if static else 0,
        online["effective_independent_n"] if online else 0,
    )
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7775.v676.independent_evidence_audit.v1",
        "experiment_id": 7775,
        "milestone": "2026.09.676",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v676_evidence"
        if blocked
        else "complete_null_independent_audit",
        "verdict_class": "blocked" if blocked else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "acceptance_gate_results": {
            gate: True if gate == "validity" else 0 if gate == "readiness" else None
            for gate in GATES
        },
        "sample_size_budget": {
            "intended": {"static_families": 64, "stream_families": 96, "retention_families": 32},
            "eligible": independent_n if static_ok and online_ok else 0,
            "started": independent_n,
            "completed": independent_n,
            "excluded": sum(bool(r["exclusions"]) for r in rows),
            "censored": sum(bool(r["censored"]) for r in rows),
            "effective_independent_n": independent_n,
        },
        "source_artifact_hashes": sources,
        "independent_static_eligible": bool(static_ok),
        "independent_online_eligible": bool(online_ok),
        "independent_findings": {
            "static": static,
            "online": online,
            "pre_gate_distinction": "A conductor receipt explains absence but supplies no scientific rows.",
        },
        "claim_scope": {
            "natural_annotations": "exposed_development_only",
            "fixture_truth": "circular_positive",
            "fresh_generalization_eligible": False,
        },
        "verifier_is_oracle": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
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
        "random_seed": {"bootstrap": 67675},
        "duration_s": 0.0,
        "phase_spans": [],
        "preconditions_checked": {
            "root": str(root.resolve()),
            "declared_paths": PLAN,
            "producer_states": {s["upstream_id"]: s["state"] for s in sources},
            "backend": "host aggregation",
            "resources": "no model load",
        },
        "validation_receipts": {
            "frozen_affected_scope": {},
            "required_commands": [],
            "terminal_readers": [],
            "e2e_checks": [],
        },
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": sources,
            "code": sha256_file(Path(__file__)),
            "seed": artifact["random_seed"],
            "roles": ["evaluation64", "retention32"],
            "parameters": {"action_costs": [5.0, 1.0, 0.25], "bootstrap_draws": 10000},
        }
    )
    artifact["field_principles"] = {key: "Exact evidence bounds this field." for key in artifact}
    artifact["field_principles"].update(
        {
            "experiment_id": "An artifact must have a unique current owner.",
            "milestone": "The claim belongs to the current milestone.",
            "run_date": "The date identifies this execution.",
            "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
            "verdict_class": "The claim class travels with the evidence.",
            "flagged_adversarial": "Invalid evidence must not open downstream gates.",
            "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
            "rows": "Aggregates must be recomputable without rerunning science.",
            "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
            "acceptance_gate_results": "A working protocol is not evidence of benefit.",
            "duration_s": "Duration must describe actual work without padding.",
            "phase_spans": "Phase times expose validation and cold replay work.",
            "random_seed": "A third party needs the same experiment inputs.",
            "reproducibility_checksum": "Exact inputs, code, roles and parameters identify the run.",
            "source_artifact_hashes": "A missing producer cannot be replaced with a convenient old result.",
            "preconditions_checked": "Access and validity must be established before expensive work.",
            "validation_receipts": "All registered checks must pass before readiness opens.",
            "verifier_is_oracle": "Execution truth and independent semantic verification are distinct.",
            "claim_scope": "Natural annotation comparison is exposed development evidence.",
            "inference_substrate": "Duration floors must match the invoked substrate.",
            "inference_substrate_class": "Aggregation uses no current model invocation.",
            "MODEL_SPECS": "An upstream model is not a current model load.",
            "model_specs": "An upstream model is not a current model load.",
            "model_invocation_counts": "Zero calls must be explicit.",
            "independent_static_eligible": "Producer self-report is insufficient for an evidence claim.",
            "independent_online_eligible": "A learning claim needs causal, static, and retention checks.",
            "independent_findings": "An audit must explain missing evidence without inventing it.",
        }
    )
    artifact["gate_principles"] = {
        "validity": "Exact custody and required validation must pass.",
        "readiness": "Complete external evidence and valid owned work are both required.",
        "probability_quality": "A probability claim requires independent proper-score reduction.",
        "decision_benefit": "A useful decision needs a paired comparator and uncertainty.",
        "retention": "New learning must preserve earlier performance.",
        "efficiency": "An efficiency claim requires measured resource use.",
    }
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """Reopen source bytes in another process and compare stable claims."""
    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    sources, failures = inspect_sources(root)
    reductions, raw_failures = read_branches(root, sources)
    rebuilt = build_artifact(
        root,
        value["run_date"],
        sources,
        failures + raw_failures,
        reductions[7772],
        reductions[7774],
    )
    keys = (
        "source_artifact_hashes",
        "gate_check_summary",
        "rows",
        "independent_findings",
        "sample_size_budget",
        "reproducibility_checksum",
    )
    return [f"{key}_changed" for key in keys if value.get(key) != rebuilt[key]]


def run_experiment(root: Path, date: str, output: Path, *, validate: bool = True) -> dict[str, Any]:
    """Freeze exact inputs, run owned checks, then publish one terminal record."""
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
    scope_path = root / SCOPE_PATH
    if not scope_path.is_file():
        scope_path = Path(__file__).parents[2] / SCOPE_PATH
    scope = json.loads(scope_path.read_bytes())
    raw_dir = root / "results/raw/experiment_7775_v676_independent_evidence_audit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources, failures = inspect_sources(root)
    reductions, raw_failures = read_branches(root, sources)
    failures.extend(raw_failures)
    finish("preconditions", len(sources))
    progress(started, "reduction", "start", 0)
    artifact = build_artifact(root, date, sources, failures, reductions[7772], reductions[7774])
    artifact["validation_receipts"]["frozen_affected_scope"] = scope
    finish("reduction", len(artifact["rows"]))
    if validate:
        private = Path(tempfile.mkdtemp(prefix="exp7775-", dir="/tmp"))
        basetemp = private / "basetemp"
        basetemp.parent.mkdir(parents=True, exist_ok=True)
        progress(started, "basetemp_probe", "start", 0)
        probe = checks.CommandSpec(
            "basetemp_parent_probe",
            (
                str(root / ".venv/bin/python"),
                "-c",
                "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); print(p.is_dir())",
                str(basetemp),
            ),
            "setup",
            30,
        )
        probe_receipts = checks.run_commands(
            root, [probe], log_dir=raw_dir / "validation/probe", heartbeat_s=30
        )
        artifact["validation_receipts"]["basetemp_parent_probe"] = probe_receipts
        finish("basetemp_probe", 1)
        progress(started, "affected_validation", "start", 0)
        commands = checks.build_scoped_commands(
            root,
            scope["test_paths"],
            scope["changed_modules"],
            static_paths=scope["static_paths"],
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
        full_base = private / "full" / "pytest"
        full_base.parent.mkdir(parents=True, exist_ok=True)
        full = checks.CommandSpec(
            "full_python_suite",
            (
                str(root / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={full_base}",
                "tests/python",
                "-q",
            ),
            "repository_health",
            1800,
        )
        full_receipts = checks.run_commands(
            root,
            [full],
            log_dir=raw_dir / "validation/full",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["full_python_suite"] = full_receipts
        artifact["validation_receipts"]["global_suite_debt"] = (
            [] if full_receipts[0]["passed"] else full_receipts
        )
        finish("full_python_suite", 1)
        if (
            not probe_receipts[0]["passed"]
            or not artifact["validation_receipts"]["required_checks_passed"]
        ):
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["acceptance_gate_results"]["validity"] = False
            artifact["acceptance_gate_results"]["readiness"] = 0
    artifact["phase_spans"] = list(spans)
    artifact["duration_s"] = time.monotonic() - started
    candidate = raw_dir / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    if validate:
        progress(started, "terminal_readers", "start", 0)
        python = str(root / ".venv/bin/python")
        readers = [
            checks.CommandSpec(
                "fresh_process_cold_replay",
                (
                    python,
                    "-u",
                    "scripts/experiments/experiment_7775_v676_independent_evidence_audit.py",
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
                "verdict_row_consistency_strict",
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
            {
                "name": "row_to_verdict_cold_replay",
                "passed": terminal[0]["passed"],
                "log_sha256": terminal[0]["log_sha256"],
            }
        ]
        try:
            report = json.loads((root / terminal[1]["log_path"]).read_text())
            artifact["flagged_adversarial"] = bool(report["flagged_count"])
        except (ValueError, KeyError, OSError):
            artifact["flagged_adversarial"] = True
        if not all(r["passed"] for r in terminal) or artifact["flagged_adversarial"]:
            artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
            artifact["verdict_class"] = "disqualified"
            artifact["acceptance_gate_results"]["validity"] = False
            artifact["acceptance_gate_results"]["readiness"] = 0
        finish("terminal_readers", len(terminal))
    progress(started, "publication", "start", 0)
    finish("publication", 1)
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    atomic_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Keep the live CLI and cold reader small enough to test separately."""
    import argparse

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
    output = args.output or root / "results/experiment_7775_v676_independent_evidence_audit.json"
    run_experiment(root, args.date, output, validate=args.fixture_root is None)
    return 0
