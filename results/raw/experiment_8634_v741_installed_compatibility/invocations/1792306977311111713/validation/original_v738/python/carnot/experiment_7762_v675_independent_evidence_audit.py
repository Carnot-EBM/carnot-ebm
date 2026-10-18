"""Reduce V675 decision and acquisition evidence from saved primitive rows.

REQ-REPORT-7762. Producer summary functions are deliberately outside this module.
The two producer contracts are checked separately so one absent input cannot
hide a present branch's failure.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file


PLAN = {
    7758: "results/experiment_7758_v675_view_decision_measurement.json",
    7761: "results/experiment_7761_v675_continuous_acquisition.json",
}
RAW_KEYS = {7758: ("raw_rows_path",), 7761: ("raw_rows_path", "event_rows_path")}
SCOPE_PATH = (
    "results/raw/experiment_7762_v675_independent_evidence_audit/frozen_affected_scope.json"
)
GATES = (
    "validity",
    "readiness",
    "probability_quality",
    "decision_benefit",
    "retention",
    "efficiency",
)


def failed(number: int, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Save both operands of a failed upstream check."""
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
    """Read only the two declared current producer paths."""
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number, relative in PLAN.items():
        path = root / relative
        source: dict[str, Any] = {
            "upstream_id": f"Exp{number}",
            "path": relative,
            "sha256": sha256_file(path) if path.is_file() else None,
            "state": "missing",
            "run_date": None,
            "imported_fields": {},
            "eligible": False,
            "pre_gate_receipt": None,
        }
        receipt_paths = sorted(root.glob(f"results/**/*{number}*pre*gate*.json"))
        if receipt_paths:
            receipt = receipt_paths[0]
            source["pre_gate_receipt"] = {
                "path": str(receipt.relative_to(root)),
                "sha256": sha256_file(receipt),
                "role": "explanation_only",
            }
        if not path.is_file():
            failures.append(
                failed(number, path, "producer_path", "existing declared producer", "missing")
            )
        else:
            try:
                value = json.loads(path.read_bytes())
                if not isinstance(value, dict):
                    raise ValueError("producer object required")
            except (ValueError, UnicodeError) as exc:
                value = {}
                failures.append(failed(number, path, "schema", "JSON object", type(exc).__name__))
            source["state"] = "present"
            source["run_date"] = value.get("run_date")
            for field, expected in (
                ("experiment_id", number),
                ("milestone", "2026.09.675"),
                ("run_date", "20260927"),
                ("flagged_adversarial", False),
            ):
                observed = value.get(field)
                source["imported_fields"][field] = observed
                if observed != expected:
                    failures.append(failed(number, path, field, expected, observed))
            for field in ("honest_verdict", "verdict_class", *RAW_KEYS[number]):
                source["imported_fields"][field] = value.get(field)
            verdict = value.get("honest_verdict")
            if not isinstance(verdict, str) or not verdict.startswith("complete_"):
                failures.append(failed(number, path, "honest_verdict", "complete_*", verdict))
            if value.get("verdict_class") not in {"positive", "null", "circular_positive"}:
                failures.append(
                    failed(
                        number,
                        path,
                        "verdict_class",
                        "positive|null|circular_positive",
                        value.get("verdict_class"),
                    )
                )
            for key in RAW_KEYS[number]:
                if not isinstance(value.get(key), str) or not value[key]:
                    failures.append(failed(number, path, key, "declared raw path", value.get(key)))
            source["eligible"] = not any(x["upstream_id"] == f"Exp{number}" for x in failures)
            source["state"] = "eligible" if source["eligible"] else "disqualified"
        sources.append(source)
    return sources, failures


def _score(row: dict[str, Any]) -> tuple[float, float, float, int]:
    """Compute proper probability loss and the fixed three-action loss."""
    p, y, action = row["probability"], row["label"], row["action"]
    if type(y) is not int or y not in (0, 1) or type(p) not in (int, float) or not 0 < p < 1:
        raise ValueError("probability")
    if action not in {"accept", "reject", "escalate"}:
        raise ValueError("action")
    brier = (p - y) ** 2
    nll = -(y * math.log(p) + (1 - y) * math.log1p(-p))
    cost = (
        0.25
        if action == "escalate"
        else 5.0
        if action == "accept" and y == 1
        else 1.0
        if action == "reject" and y == 0
        else 0.0
    )
    return brier, nll, cost, int(action == "accept" and y == 1)


def _paired(values: list[float], seed: int) -> dict[str, Any]:
    """Bootstrap independent families and compute a two-sided sign test."""
    n = len(values)
    if not n:
        return {"mean": None, "lower95": None, "upper95": None, "p": None, "n": 0}
    rng = random.Random(seed)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    nonzero = [x for x in values if x != 0]
    k = min(sum(x > 0 for x in nonzero), sum(x < 0 for x in nonzero))
    p = (
        min(1.0, 2 * sum(math.comb(len(nonzero), i) for i in range(k + 1)) / 2 ** len(nonzero))
        if nonzero
        else 1.0
    )
    return {"mean": sum(values) / n, "lower95": draws[250], "upper95": draws[9749], "p": p, "n": n}


def _holm(tests: dict[str, dict[str, Any]]) -> None:
    """Attach monotone Holm adjusted probabilities to paired tests."""
    ordered = sorted(tests, key=lambda key: tests[key]["p"])
    running = 0.0
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, tests[key]["p"] * (len(ordered) - index)))
        tests[key]["holm_p"] = running


def reduce_static(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    """Average views and seeds inside a family before paired inference."""
    errors: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scored: dict[tuple[str, str], list[tuple[float, float, float, int]]] = defaultdict(list)
    arms, seeds, views = summary["arms"], summary["seeds"], summary["views"]
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
            scored[(family, row["arm"])].append(_score(row))
        except (ValueError, KeyError):
            errors.add("probability")
    if len(groups) != summary["expected_families"]:
        errors.add("family_count")
    expected = {(arm, seed, view) for arm in arms for seed in seeds for view in views}
    for family, family_rows in groups.items():
        if (
            len(family_rows) != len(expected)
            or {(r["arm"], r["seed"], r["view"]) for r in family_rows} != expected
        ):
            errors.add("roster")
        if len({r["source_sha256"] for r in family_rows}) != 1:
            errors.add("source_identity")
        erased = {r["input_hash"] for r in family_rows if r["arm"] == "source_erased"}
        original = {r["input_hash"] for r in family_rows if r["arm"] != "source_erased"}
        if erased & original:
            errors.add("erased_boundary")
        if len({r["label"] for r in family_rows}) != 1:
            errors.add("label_join")
    by_arm: dict[str, dict[str, float]] = {}
    family_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (family, arm), values in scored.items():
        if values:
            family_metrics[family][arm] = {
                "brier": sum(v[0] for v in values) / len(values),
                "nll": sum(v[1] for v in values) / len(values),
                "cost": sum(v[2] for v in values) / len(values),
                "false_accepts": sum(v[3] for v in values) / len(values),
            }
    for arm in arms:
        values = [metrics[arm] for metrics in family_metrics.values() if arm in metrics]
        if values:
            by_arm[arm] = {key: sum(v[key] for v in values) / len(values) for key in values[0]}
    tests: dict[str, dict[str, Any]] = {}
    for comparator in ("augmented_set", "constrained_mlp", "local_logistic", "source_erased"):
        if comparator not in arms or "constrained_set" not in arms:
            continue
        for metric in ("cost", "brier"):
            deltas = [
                metrics[comparator][metric] - metrics["constrained_set"][metric]
                for metrics in family_metrics.values()
                if comparator in metrics and "constrained_set" in metrics
            ]
            tests[f"{comparator}_{metric}"] = _paired(deltas, 67562)
    if tests:
        _holm(tests)
    return {
        "failed_checks": sorted(errors),
        "effective_independent_n": len(groups),
        "by_arm": by_arm,
        "paired_tests": tests,
        "coverage": sum(r["action"] != "escalate" for r in rows) / len(rows) if rows else None,
        "rows": rows,
    }


def reduce_online(
    rows: list[dict[str, Any]], events: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    """Replay feedback and admission order against saved predictions."""
    errors: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scored: dict[str, list[tuple[float, float, float, int]]] = defaultdict(list)
    event_keys = {(e.get("kind"), e.get("arm"), e.get("family_id"), e.get("tick")) for e in events}
    for row in rows:
        groups[row["family_id"]].append(row)
        if row.get("label_join") != row["family_id"]:
            errors.add("label_join")
        if row.get("role") != "evaluation64":
            errors.add("role")
        if row["prediction_tick"] >= row["feedback_tick"]:
            errors.add("feedback_chronology")
        for kind, tick_key in (("prediction", "prediction_tick"), ("feedback", "feedback_tick")):
            if (kind, row["arm"], row["family_id"], row[tick_key]) not in event_keys:
                errors.add("event_join")
        try:
            scored[row["arm"]].append(_score(row))
        except (ValueError, KeyError):
            errors.add("probability")
    if len(groups) != summary["expected_families"]:
        errors.add("family_count")
    for group in groups.values():
        if len(group) != len(summary["arms"]) or {r["arm"] for r in group} != set(summary["arms"]):
            errors.add("roster")
        if len({r["source_sha256"] for r in group}) != 1 or len({r["label"] for r in group}) != 1:
            errors.add("source_identity")
    static = summary["static_dictionary"]
    if (
        len(static) != 16
        or set(static) != set(summary["complete_static_predicates"])
        or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in static.values())
    ):
        errors.add("static_closure")
    admissions = [e for e in events if e.get("kind") == "admission"]
    if len({e["feedback_id"] for e in admissions}) != len(admissions):
        errors.add("one_use_admission")
    if len(admissions) > 8 or summary["proposal_count"] > 8:
        errors.add("proposal_credits")
    if summary["pending_high_water"] > 12:
        errors.add("pending_capacity")
    if summary["restart_exact_parity"] is not True:
        errors.add("restart_parity")
    feedback_ticks = {
        e["family_id"]: e["tick"]
        for e in events
        if e.get("kind") == "feedback" and e.get("arm") == "adaptive"
    }
    later = {e["feedback_id"]: e for e in events if e.get("kind") == "later_prediction"}
    causal_changes = 0
    for admission in admissions:
        fid = admission["feedback_id"]
        witness = later.get(fid)
        if (
            fid not in feedback_ticks
            or admission["tick"] <= feedback_ticks[fid]
            or witness is None
            or witness["tick"] <= admission["tick"]
        ):
            errors.add("update_chronology")
        elif witness["probability"] != witness["erased_probability"]:
            causal_changes += 1
    retention = summary["retention_rows"]
    if retention and any(r.get("role") != "retention32" for r in retention):
        errors.add("retention_role")
    by_arm = {
        arm: {
            "brier": sum(v[0] for v in values) / len(values),
            "cost": sum(v[2] for v in values) / len(values),
            "false_accepts": sum(v[3] for v in values),
        }
        for arm, values in scored.items()
        if values
    }
    return {
        "failed_checks": sorted(errors),
        "effective_independent_n": len(groups),
        "by_arm": by_arm,
        "admission_count": len(admissions),
        "causal_changes": causal_changes,
        "retention_count": len(retention),
        "rows": rows,
        "events": events,
    }


def read_branches(
    root: Path, sources: list[dict[str, Any]]
) -> tuple[dict[int, Any], list[dict[str, Any]], dict[str, str]]:
    """Open qualifying raw files and verify each declared byte hash."""
    reductions: dict[int, Any] = {7758: None, 7761: None}
    failures: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for number, relative in PLAN.items():
        if not next(source for source in sources if source["upstream_id"] == f"Exp{number}")[
            "eligible"
        ]:
            continue
        producer = json.loads((root / relative).read_bytes())
        payloads = []
        for key in RAW_KEYS[number]:
            raw = (root / producer[key]).resolve()
            if not raw.is_relative_to(root.resolve()) or not raw.is_file():
                failures.append(
                    failed(number, raw, key, "existing raw file below root", producer[key])
                )
                continue
            digest = sha256_file(raw)
            hashes[str(raw)] = digest
            declared = producer.get(key.removesuffix("_path") + "_sha256")
            if declared != digest:
                failures.append(failed(number, raw, key + "_sha256", digest, declared))
            try:
                payloads.append(json.loads(raw.read_bytes()))
            except (ValueError, UnicodeError) as exc:
                failures.append(failed(number, raw, "raw_schema", "JSON", type(exc).__name__))
        if len(payloads) != len(RAW_KEYS[number]) or any(
            x["upstream_id"] == f"Exp{number}" for x in failures
        ):
            continue
        try:
            reductions[number] = (
                reduce_static(payloads[0]["rows"], payloads[0]["summary"])
                if number == 7758
                else reduce_online(payloads[0]["rows"], payloads[1], payloads[0]["summary"])
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            failures.append(
                failed(
                    number,
                    root / producer["raw_rows_path"],
                    "raw_schema",
                    "reducible",
                    type(exc).__name__,
                )
            )
            continue
        for check in reductions[number]["failed_checks"]:
            failures.append(
                failed(number, root / producer["raw_rows_path"], "raw_rows", "valid", check)
            )
    return reductions, failures, hashes


def build_artifact(
    root: Path,
    date: str,
    sources: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    static: dict[str, Any] | None,
    online: dict[str, Any] | None,
    raw_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a terminal audit record even when every producer is absent."""
    source_failures = {f["upstream_id"] for f in failures}
    static_ok = (
        static is not None and "Exp7758" not in source_failures and not static["failed_checks"]
    )
    online_ok = (
        online is not None
        and "Exp7761" not in source_failures
        and not online["failed_checks"]
        and online["causal_changes"] > 0
        and online["retention_count"] > 0
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
    for number, reduction in ((7758, static), (7761, online)):
        if reduction is not None:
            source = next(s for s in sources if s["upstream_id"] == f"Exp{number}")
            for row in reduction["rows"]:
                try:
                    brier, _nll, cost, _false_accept = _score(row)
                    metrics = {"probability": row["probability"], "brier": brier, "cost": cost}
                except (KeyError, ValueError):
                    metrics = None
                rows.append(
                    {
                        "family_id": row["family_id"],
                        "arm": row["arm"],
                        "seed": row["seed"],
                        "raw_path": source["imported_fields"].get("raw_rows_path"),
                        "raw_hash": (raw_hashes or {}).get(
                            str(
                                (
                                    root / source["imported_fields"].get("raw_rows_path", "")
                                ).resolve()
                            )
                        ),
                        "metrics": metrics,
                        "label": row["label"],
                        "exclusions": [] if metrics is not None else ["invalid_metric"],
                        "censored": row.get("censored", False),
                    }
                )
    independent_n = max(
        static["effective_independent_n"] if static else 0,
        online["effective_independent_n"] if online else 0,
    )
    blocked = bool(failures or static is None or online is None)
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7762.v675.independent_evidence_audit.v1",
        "experiment_id": 7762,
        "milestone": "2026.09.675",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v675_evidence"
        if blocked
        else "complete_null_independent_audit",
        "verdict_class": "blocked" if blocked else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "acceptance_gate_results": {
            key: True if key == "validity" else 0 if key == "readiness" else None for key in GATES
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
        "raw_input_hashes": raw_hashes or {},
        "independent_static_eligible": bool(static_ok),
        "independent_online_eligible": bool(online_ok),
        "independent_findings": {
            "static": {
                "status": "eligible"
                if static_ok
                else "blocked"
                if static is None
                else "disqualified",
                "reduction": static,
            },
            "online": {
                "status": "eligible"
                if online_ok
                else "blocked"
                if online is None
                else "disqualified",
                "reduction": online,
            },
            "pre_gate_distinction": "A conductor receipt cannot substitute for a producer or raw scientific rows.",
        },
        "recomputed_static": static,
        "recomputed_online": online,
        "claim_scope": {
            "RAGTruth": "exposed_development_only",
            "constructed_truth": "fixture_only",
            "source_erased": "required_control",
            "fresh_generalization_eligible": False,
        },
        "verifier_is_oracle": root.resolve() != Path(__file__).parents[2].resolve(),
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
        "random_seed": {"bootstrap": 67562},
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
            "raw": artifact["raw_input_hashes"],
            "code": sha256_file(Path(__file__)),
            "seed": artifact["random_seed"],
        }
    )
    artifact["field_principles"] = {key: "Exact evidence bounds this field." for key in artifact}
    artifact["field_principles"].update(
        {
            "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
            "verdict_class": "The claim class travels with the evidence.",
            "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
            "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
            "acceptance_gate_results": "A working protocol is not evidence of benefit.",
            "source_artifact_hashes": "A missing producer cannot be replaced with a convenient old result.",
            "independent_findings": "An audit must explain missing evidence without inventing it.",
        }
    )
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """Reopen exact upstream bytes and rebuild claims in this process."""
    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    sources, failures = inspect_sources(root)
    reductions, raw_failures, hashes = read_branches(root, sources)
    rebuilt = build_artifact(
        root,
        value["run_date"],
        sources,
        failures + raw_failures,
        reductions[7758],
        reductions[7761],
        hashes,
    )
    return [
        f"{key}_changed"
        for key in (
            "source_artifact_hashes",
            "raw_input_hashes",
            "gate_check_summary",
            "rows",
            "recomputed_static",
            "recomputed_online",
            "independent_findings",
            "sample_size_budget",
            "reproducibility_checksum",
        )
        if value.get(key) != rebuilt[key]
    ]


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Emit a flushed monotonic boundary for a supervised phase."""
    import time

    print(
        f"[exp7762] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def run_experiment(root: Path, date: str, output: Path, *, validate: bool = True) -> dict[str, Any]:
    """Freeze scope, run checks, then publish one terminal result atomically."""
    import tempfile
    import time

    from carnot.reporting.current_work_receipt import atomic_json
    from carnot.reporting import experiment_7303_validation_scope as checks

    root = root.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    phase_start = 0.0

    def finish(phase: str, units: int) -> None:
        nonlocal phase_start
        end = time.monotonic() - started
        spans.append(
            {
                "phase": phase,
                "start_s": phase_start,
                "end_s": end,
                "duration_s": end - phase_start,
                "completed_units": units,
            }
        )
        phase_start = end
        progress(started, phase, "complete", units)

    progress(started, "preconditions", "start", 0)
    scope = (
        json.loads((root / SCOPE_PATH).read_bytes())
        if (root / SCOPE_PATH).is_file()
        else json.loads((Path(__file__).parents[2] / SCOPE_PATH).read_bytes())
    )
    raw_dir = root / "results/raw/experiment_7762_v675_independent_evidence_audit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources, failures = inspect_sources(root)
    reductions, raw_failures, hashes = read_branches(root, sources)
    failures.extend(raw_failures)
    finish("preconditions", len(sources))
    progress(started, "reduction", "start", 0)
    artifact = build_artifact(
        root, date, sources, failures, reductions[7758], reductions[7761], hashes
    )
    artifact["validation_receipts"]["frozen_affected_scope"] = scope
    finish("reduction", len(artifact["rows"]))
    if validate:
        private = Path(tempfile.mkdtemp(prefix="exp7762-", dir="/tmp"))
        basetemp = private / "basetemp"
        basetemp.parent.mkdir(parents=True, exist_ok=True)
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
        progress(started, "basetemp_probe", "start", 0)
        probe_receipt = checks.run_commands(
            root, [probe], log_dir=raw_dir / "validation/probe", heartbeat_s=30
        )
        artifact["validation_receipts"]["basetemp_parent_probe"] = probe_receipt
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
        full_receipt = checks.run_commands(
            root,
            [full],
            log_dir=raw_dir / "validation/full",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["full_python_suite"] = full_receipt
        artifact["validation_receipts"]["global_suite_debt"] = (
            [] if full_receipt[0]["passed"] else full_receipt
        )
        finish("full_python_suite", 1)
        if (
            not probe_receipt[0]["passed"]
            or not artifact["validation_receipts"]["required_checks_passed"]
        ):
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["acceptance_gate_results"]["validity"] = False
            artifact["acceptance_gate_results"]["readiness"] = 0
    artifact["phase_spans"] = deepcopy(spans)
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
                    "scripts/experiments/experiment_7762_v675_independent_evidence_audit.py",
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
        artifact["flagged_adversarial"] = not terminal[1]["passed"]
        if not all(r["passed"] for r in terminal):
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
    """Expose a small live entrypoint and fresh-process reader."""
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
    output = args.output or root / "results/experiment_7762_v675_independent_evidence_audit.json"
    run_experiment(root, args.date, output, validate=args.fixture_root is None)
    return 0
