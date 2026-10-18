"""Independent V674 row and event reduction for REQ-REPORT-7747.

The reducer reads saved primitive observations. It does not call a producer's
summary functions, so an incorrect producer summary can be found independently.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file


STATIC_ARMS = (
    "local_set",
    "response_set",
    "local_mlp",
    "local_logistic",
    "source_erased",
    "pooled_mlp",
    "pooled_logistic",
)
ONLINE_ARMS = ("adaptive", "frozen", "complete_static", "shuffled")
PLAN = {
    7744: "results/experiment_7744_v674_decision_measurement.json",
    7746: "results/experiment_7746_v674_continuous_acquisition.json",
}
HISTORY = {
    7734: "results/experiment_7734_v673_independent_evidence_audit.json",
    7738: "results/experiment_7738_v673_capstone.json",
    7745: "results/experiment_7745_v674_qwen_localization.json",
}
SCOPE = {
    "test_paths": [
        "tests/python/test_experiment_7747_v674_independent_evidence_audit.py",
        "tests/python/test_experiment_7742_v674_bank_qualification.py",
        "tests/python/test_experiment_7740_v674_sentence_label_protocol.py",
        "tests/python/test_experiment_7734_v673_independent_evidence_audit.py",
    ],
    "changed_modules": ["python/carnot/experiment_7747_v674_independent_evidence_audit.py"],
    "static_paths": ["scripts/experiments/experiment_7747_v674_independent_evidence_audit.py"],
    "requirements": ["REQ-REPORT-7747", "REQ-CL-7747-AUDIT"],
}


def failed(
    check: str, upstream_id: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep exact operands so a missing input is a checkable contract failure."""
    return {
        "check": check,
        "upstream_id": upstream_id,
        "artifact_path": str(path.resolve()),
        "field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def inspect_sources(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Separate eligible producers, historical records, and absent inputs."""
    import json

    rows: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        key: {}
        for key in (
            "eligible_producers",
            "historical_disqualified_sources",
            "missing_inputs",
            "pre_gate_receipts",
        )
    }
    failures: list[dict[str, Any]] = []
    for number, relative in {**PLAN, **HISTORY}.items():
        path = root / relative
        required = number in PLAN
        state = "missing"
        value: dict[str, Any] = {}
        if path.is_file():
            try:
                parsed = json.loads(path.read_bytes())
                if isinstance(parsed, dict):
                    value = parsed
            except (ValueError, UnicodeError):
                pass
            verdict = value.get("honest_verdict")
            if (
                isinstance(verdict, str)
                and verdict.startswith("complete_")
                and value.get("verdict_class") in {"positive", "null", "circular_positive"}
                and value.get("flagged_adversarial") is False
            ):
                state = "eligible"
            else:
                state = "ineligible"
        digest = sha256_file(path) if path.is_file() else None
        bucket = (
            "eligible_producers"
            if state == "eligible" and required
            else "historical_disqualified_sources"
            if not required and state != "missing"
            else "pre_gate_receipts"
            if state == "ineligible"
            else "missing_inputs"
        )
        hashes[bucket][relative] = digest
        rows.append(
            {
                "upstream_id": f"Exp{number}",
                "artifact_path": relative,
                "state": state,
                "sha256": digest,
                "verdict_class": value.get("verdict_class"),
            }
        )
        if required and state != "eligible":
            failures.append(
                failed(
                    "required_source_eligible",
                    f"Exp{number}",
                    path,
                    "verdict_class",
                    "positive|null|circular_positive with clean complete verdict",
                    value.get("verdict_class") if state != "missing" else "missing",
                )
            )
    return rows, hashes, failures


def _metric(row: dict[str, Any]) -> tuple[float, float, float]:
    """Recompute probability loss and fixed action cost from the saved label."""
    p, y = row["probability"], row["label"]
    if type(y) is not int or y not in (0, 1) or not isinstance(p, (int, float)) or not 0 < p < 1:
        raise ValueError("invalid probability or binary label")
    brier = (p - y) ** 2
    nll = -(y * math.log(p) + (1 - y) * math.log1p(-p))
    action = row["action"]
    if action not in {"accept", "reject", "escalate"}:
        raise ValueError("invalid action")
    cost = (
        0.25
        if action == "escalate"
        else 5.0
        if action == "accept" and y == 0
        else 1.0
        if action == "reject" and y == 1
        else 0.0
    )
    return brier, nll, cost


def reduce_static(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    """Count families once while checking every paired arm and fixed score."""
    errors: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    means: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        groups[row["family_id"]].append(row)
        if row["features_sha256"] == row["annotation_sha256"]:
            errors.add("feature_annotation_separation")
        if not row["sentence_offsets_valid"]:
            errors.add("sentence_offsets")
        if not row["prediction_before_label"]:
            errors.add("label_leakage")
        if row["optimizer_steps"] != summary["optimizer_steps"]:
            errors.add("optimizer_budget")
        if row["temperature"] != summary["temperature"]:
            errors.add("temperature_selection")
        try:
            scores = _metric(row)
        except ValueError:
            errors.add("metric_arithmetic")
            continue
        if any(
            not math.isclose(row[key], score, abs_tol=1e-8)
            for key, score in zip(("brier", "nll", "cost"), scores, strict=True)
        ):
            errors.add("metric_arithmetic")
        means[row["arm"]].append(scores)
    for group in groups.values():
        if {(r["arm"], r["seed"]) for r in group} != {
            (arm, seed) for arm in STATIC_ARMS for seed in range(5)
        } or len(group) != 35:
            errors.add("arm_roster")
        if (
            len({r["source_sha256"] for r in group}) != 1
            or len({r["input_hash"] for r in group}) != 1
        ):
            errors.add("source_identity")
    by_arm = {
        arm: {
            name: sum(item[index] for item in values) / len(values)
            for index, name in enumerate(("brier", "nll", "cost"))
        }
        for arm, values in means.items()
        if values
    }
    pooled = sum(item[0] for values in means.values() for item in values) / max(
        1, sum(map(len, means.values()))
    )
    if not math.isclose(pooled, summary["pooled_brier"], abs_tol=1e-8):
        errors.add("pooled_metric")
    return {
        "failed_checks": sorted(errors),
        "effective_independent_n": len(groups),
        "by_arm": by_arm,
        "pooled_brier": pooled,
        "rows": rows,
    }


def reduce_online(
    rows: list[dict[str, Any]], events: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    """Replay the delayed order and require a later prediction change for learning."""
    errors: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scores: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        groups[row["family_id"]].append(row)
        if row["prediction_tick"] >= row["feedback_tick"]:
            errors.add("future_feedback")
        try:
            measured = _metric(row)
        except ValueError:
            errors.add("metric_arithmetic")
            continue
        if not math.isclose(row["brier"], measured[0], abs_tol=1e-8) or not math.isclose(
            row["cost"], measured[2], abs_tol=1e-8
        ):
            errors.add("metric_arithmetic")
        scores[row["arm"]].append(measured)
    for group in groups.values():
        if {r["arm"] for r in group} != set(ONLINE_ARMS) or len(group) != len(ONLINE_ARMS):
            errors.add("arm_roster")
        if (
            len({r["source_sha256"] for r in group}) != 1
            or len({r["input_hash"] for r in group}) != 1
        ):
            errors.add("source_identity")
    event_keys = {
        (e["family_id"], e["arm"], e["kind"], e["tick"])
        for e in events
        if e["kind"] in {"prediction", "feedback"}
    }
    for row in rows:
        if (
            row["family_id"],
            row["arm"],
            "prediction",
            row["prediction_tick"],
        ) not in event_keys or (
            row["family_id"],
            row["arm"],
            "feedback",
            row["feedback_tick"],
        ) not in event_keys:
            errors.add("event_chronology")
    admissions = [e for e in events if e["kind"] == "admission"]
    if len({e["feedback_id"] for e in admissions}) != len(admissions):
        errors.add("one_use_admission")
    if len(admissions) > 8 or summary["proposal_count"] > 8:
        errors.add("proposal_credits")
    if summary["pending_high_water"] > 12:
        errors.add("pending_capacity")
    if len(summary["static_dictionary"]) != 16 or any(
        not isinstance(weight, (int, float)) or not math.isfinite(weight)
        for weight in summary["static_dictionary"].values()
    ):
        errors.add("static_closure")
    if summary["restart_exact_parity"] is not True:
        errors.add("restart_parity")
    if summary["consumed_exp7744_evaluation"] is not False:
        errors.add("evaluation_isolation")
    for event in events:
        if event["kind"] == "shuffled_label" and event["origin_feedback_tick"] >= event["tick"]:
            errors.add("future_origin_shuffled_label")
        if event["kind"] == "admission" and math.isclose(
            event["later_probability"], event["erased_probability"], abs_tol=1e-12
        ):
            errors.add("causal_erasure")
    return {
        "failed_checks": sorted(errors),
        "effective_independent_n": len(groups),
        "by_arm": {
            arm: {
                "brier": sum(s[0] for s in values) / len(values),
                "cost": sum(s[2] for s in values) / len(values),
            }
            for arm, values in scores.items()
            if values
        },
        "admission_count": len(admissions),
        "rows": rows,
        "events": events,
    }


def read_raw_sources(
    root: Path, custody: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, str], list[dict[str, Any]]]:
    """Read raw producer bytes only after the producer itself qualifies."""
    import json

    states = {item["upstream_id"]: item["state"] for item in custody}
    reductions: dict[int, dict[str, Any] | None] = {7744: None, 7746: None}
    hashes: dict[str, str] = {}
    failures: list[dict[str, Any]] = []
    for number, relative in PLAN.items():
        if states.get(f"Exp{number}") != "eligible":
            continue
        producer = json.loads((root / relative).read_bytes())
        keys = ("raw_rows_path",) if number == 7744 else ("raw_rows_path", "event_rows_path")
        paths = []
        for key in keys:
            raw_path = root / producer.get(key, "")
            if not producer.get(key) or not raw_path.is_file():
                failures.append(
                    failed(
                        "raw_input_exists",
                        f"Exp{number}",
                        raw_path,
                        key,
                        "existing file",
                        producer.get(key),
                    )
                )
            else:
                paths.append(raw_path)
                hashes[str(raw_path.resolve())] = sha256_file(raw_path)
        if len(paths) != len(keys):
            continue
        try:
            payload = json.loads(paths[0].read_bytes())
            if number == 7744:
                reductions[number] = reduce_static(payload["rows"], payload["summary"])
            else:
                events = json.loads(paths[1].read_bytes())
                reductions[number] = reduce_online(payload["rows"], events, payload["summary"])
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            failures.append(
                failed(
                    "raw_schema", f"Exp{number}", paths[0], "rows", "reducible", type(exc).__name__
                )
            )
            continue
        for check in reductions[number]["failed_checks"]:
            failures.append(failed(check, f"Exp{number}", paths[0], "raw_rows", "valid", check))
    return reductions[7744], reductions[7746], hashes, failures


def make_artifact(
    root: Path,
    date: str,
    custody: list[dict[str, Any]],
    hashes: dict[str, Any],
    failures: list[dict[str, Any]],
    static: dict[str, Any] | None,
    online: dict[str, Any] | None,
    raw_hashes: dict[str, str],
) -> dict[str, Any]:
    """Finish source accounting even when scientific measurements cannot run."""
    import os
    import socket

    valid_static = static is not None and not static["failed_checks"]
    valid_online = online is not None and not online["failed_checks"]
    eligible = valid_static and valid_online and not failures
    rows = []
    for item in custody:
        rows.append(
            {
                "family_id": item["upstream_id"],
                "arm": "custody",
                "seed": None,
                "raw_numerators": None,
                "denominators": None,
                "exclusions": [] if item["state"] == "eligible" else [item["state"]],
                "censored": False,
                "input_hash": item["sha256"],
            }
        )
    for reduction in (static, online):
        if reduction:
            rows.extend(
                {
                    "family_id": row["family_id"],
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "raw_numerators": {
                        key: row[key] for key in ("label", "probability", "brier", "cost")
                    },
                    "denominators": {"family": 1},
                    "exclusions": row["exclusions"],
                    "censored": row["censored"],
                    "input_hash": row["input_hash"],
                }
                for row in reduction["rows"]
            )
    n = max(
        static["effective_independent_n"] if static else 0,
        online["effective_independent_n"] if online else 0,
    )
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7747.v674.independent_evidence_audit.v1",
        "experiment_id": 7747,
        "milestone": "2026.09.674",
        "run_date": date,
        "honest_verdict": "complete_null_no_registered_benefit"
        if eligible
        else "complete_blocked_required_v674_evidence",
        "verdict_class": "null" if eligible else "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": {
            "validity": True,
            "readiness": None,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "rows": rows,
        "sample_size_budget": {
            "intended": {"static_families": 64, "stream_families": 96, "retention_families": 32},
            "started": n,
            "completed": n,
            "eligible": n if eligible else 0,
            "excluded": 0,
            "censored": 0,
            "effective_independent_n": n,
        },
        "claim_scope": {
            "RAGTruth": "development_only",
            "constructed_truth": "fixture_only",
            "ARC": "adapter_withheld_public",
            "fresh_generalization_eligible": False,
        },
        "fresh_generalization_eligible": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
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
        "execution_venue": "host",
        "execution_venue_details": {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "gpu_uuid": None,
        },
        "phase_spans": [],
        "duration_s": 0.0,
        "random_seed": {
            "bootstrap": 67447,
            "permutation": 67448,
            "purpose": "family-level audit replay",
        },
        "source_artifact_hashes": hashes,
        "raw_input_hashes": raw_hashes,
        "preconditions_checked": {
            "root": str(root.resolve()),
            "required_paths": PLAN,
            "schemas_checked": {x["upstream_id"]: x["state"] for x in custody},
            "effective_coding_backend": "Codex GPT-6 API session",
        },
        "validation_receipts": {
            "frozen_affected_scope": {},
            "required_commands": [],
            "terminal_readers": [],
            "e2e_checks": [],
            "global_suite_debt": [],
        },
        "verifier_is_oracle": False,
        "independent_audit_complete_score": 1,
        "independent_static_eligible": valid_static
        and not any(x["upstream_id"] == "Exp7744" for x in failures),
        "independent_online_eligible": valid_online
        and not any(x["upstream_id"] == "Exp7746" for x in failures),
        "recomputed_static": static,
        "recomputed_online": online,
        "claim_matrix": {
            item["upstream_id"]: {
                "path": item["artifact_path"],
                "hash": item["sha256"],
                "state": item["state"],
                "denominator": None,
                "recalculated_outcome": None,
                "continuation_bound": "qualify required source rows",
            }
            for item in custody
        },
        "upstream_dispositions": custody,
        "pilot_qwen_diagnostic": {
            "upstream_id": "Exp7745",
            "current_model_calls": 0,
            "state": next(x["state"] for x in custody if x["upstream_id"] == "Exp7745"),
        },
        "activation": False,
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": hashes,
            "raw": raw_hashes,
            "seed": artifact["random_seed"],
            "reducer_code": sha256_file(Path(__file__)),
        }
    )
    artifact["field_principles"] = {key: "Measured evidence bounds this claim." for key in artifact}
    artifact["field_principles"].update(
        {key: "Unmeasured is not a zero effect." for key in artifact["acceptance_gate_results"]}
    )
    artifact["field_principles"]["field_principles"] = "The rationale travels with the contract."
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """Reopen every source and recompute reductions in the reader's process."""
    import json

    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    custody, hashes, failures = inspect_sources(root)
    static, online, raw_hashes, raw_failures = read_raw_sources(root, custody)
    errors = []
    for field, observed in (
        ("upstream_dispositions", custody),
        ("source_artifact_hashes", hashes),
        ("gate_check_summary", failures + raw_failures),
        ("recomputed_static", static),
        ("recomputed_online", online),
    ):
        if value[field] != observed:
            errors.append(
                f"{field}_changed"
                if field not in {"recomputed_static", "recomputed_online"}
                else f"{field.removeprefix('recomputed_')}_reduction_changed"
            )
    if value["raw_input_hashes"] != raw_hashes:
        errors.append("raw_input_changed")
    checksum = canonical_hash(
        {
            "sources": hashes,
            "raw": raw_hashes,
            "seed": value["random_seed"],
            "reducer_code": sha256_file(Path(__file__)),
        }
    )
    if checksum != value["reproducibility_checksum"]:
        errors.append("reproducibility_checksum_changed")
    return errors


def progress(started: float, phase: str, event: str, units: int = 0) -> None:
    """Show a real task boundary before any command can run for minutes."""
    import time

    print(
        f"[exp7747] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def run_experiment(
    root: Path, date: str, output: Path, *, private_fixture: bool = False
) -> dict[str, Any]:
    """Freeze custody, run owned checks, and publish one terminal artifact."""
    import tempfile
    import time

    from carnot.reporting.current_work_receipt import atomic_json
    from carnot.reporting import experiment_7303_validation_scope as checks

    root = root.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    last = 0.0

    def span(name: str, units: int) -> None:
        nonlocal last
        end = time.monotonic() - started
        spans.append(
            {
                "phase": name,
                "run_date": date,
                "start_s": last,
                "end_s": end,
                "duration_s": end - last,
                "heartbeat_timestamps_s": [end],
                "completed_units": units,
                "checkpoint_hash": canonical_hash({"phase": name, "units": units}),
            }
        )
        last = end

    progress(started, "preconditions", "start")
    raw = root / "results/raw/experiment_7747_v674_independent_evidence_audit"
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "frozen_affected_scope.json", SCOPE)
    custody, hashes, failures = inspect_sources(root)
    static, online, raw_hashes, raw_failures = read_raw_sources(root, custody)
    failures.extend(raw_failures)
    atomic_json(
        raw / "checkpoint.json",
        {"completed_units": len(custody), "source_hashes": hashes, "raw_input_hashes": raw_hashes},
    )
    progress(started, "preconditions", "complete", len(custody))
    span("preconditions", len(custody))
    progress(started, "reduction", "start")
    artifact = make_artifact(root, date, custody, hashes, failures, static, online, raw_hashes)
    artifact["validation_receipts"]["frozen_affected_scope"] = SCOPE
    if private_fixture and not failures:
        artifact["honest_verdict"] = "complete_circular_positive_private_fixture"
        artifact["verdict_class"] = "circular_positive"
        artifact["verifier_is_oracle"] = True
    atomic_json(raw / "independent_reduction.json", {"static": static, "online": online})
    progress(started, "reduction", "complete", len(artifact["rows"]))
    span("reduction", len(artifact["rows"]))
    if private_fixture:
        artifact["phase_spans"] = spans
        artifact["duration_s"] = last
        atomic_json(output, artifact)
        progress(started, "private_fixture", "complete", 1)
        return artifact

    private = Path(tempfile.mkdtemp(prefix="exp7747-", dir="/tmp"))
    (private / "basetemp").mkdir()
    progress(started, "affected_validation", "start")
    commands = checks.build_scoped_commands(
        root,
        SCOPE["test_paths"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "basetemp",
        coverage_file=private / "coverage.data",
    )
    receipts = checks.run_commands(
        root,
        commands,
        log_dir=raw / "validation/affected",
        extra_env={"COVERAGE_FILE": str(private / "coverage.data"), "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )
    artifact["validation_receipts"]["required_commands"] = receipts
    artifact["validation_receipts"].update(checks.reduce_required_checks(receipts))
    progress(started, "affected_validation", "complete", len(receipts))
    span("affected_validation", len(receipts))
    progress(started, "global_suite", "start")
    full = checks.CommandSpec(
        "full_python_suite",
        (
            str(root / ".venv/bin/pytest"),
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={private / 'full'}",
            "tests/python",
            "-q",
        ),
        "repository_health",
        1200,
    )
    artifact["validation_receipts"]["global_suite_debt"] = checks.run_commands(
        root,
        [full],
        log_dir=raw / "validation/full",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )
    progress(started, "global_suite", "complete", 1)
    span("global_suite", 1)
    if not artifact["validation_receipts"]["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
        artifact["acceptance_gate_results"]["validity"] = False
    artifact["phase_spans"] = spans.copy()
    artifact["duration_s"] = last
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    python = str(root / ".venv/bin/python")
    terminal = [
        checks.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                "scripts/experiments/experiment_7747_v674_independent_evidence_audit.py",
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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            180,
        ),
    ]
    progress(started, "terminal_readers", "start")
    readers = checks.run_commands(
        root,
        terminal,
        log_dir=raw / "validation/terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )
    artifact["validation_receipts"]["terminal_readers"] = readers
    artifact["validation_receipts"]["exact_candidate_sha256"] = sha256_file(candidate)
    artifact["validation_receipts"]["e2e_checks"] = [
        {
            "name": "cold_cli_replay_from_raw",
            "passed": readers[0]["passed"],
            "log_sha256": readers[0]["log_sha256"],
        }
    ]
    artifact["flagged_adversarial"] = not readers[1]["passed"]
    if not all(row["passed"] for row in readers):
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
        artifact["acceptance_gate_results"]["validity"] = False
    progress(started, "terminal_readers", "complete", len(readers))
    span("terminal_readers", len(readers))
    progress(started, "publication", "start")
    span("publication", 1)
    artifact["phase_spans"] = spans
    artifact["duration_s"] = last
    atomic_json(output, artifact)
    progress(started, "publication", "complete", 1)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Keep the executable surface small and expose a fresh-process reader."""
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    if args.cold:
        errors = cold_replay(args.cold)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    root = args.fixture_root or Path(__file__).resolve().parents[2]
    output = args.output or root / "results/experiment_7747_v674_independent_evidence_audit.json"
    run_experiment(root, args.date, output, private_fixture=args.fixture_root is not None)
    return 0
