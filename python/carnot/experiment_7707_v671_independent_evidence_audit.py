"""Publish the bounded V671 independent evidence audit (REQ-REPORT-7707)."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as checks
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_audit_v671 import (
    audit_inventory,
    challenge_boundaries,
    reduce_decisions,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7707_v671_independent_evidence_audit")
PLAN = [
    (7699, "results/experiment_7699_v671_contract_methods.json", None),
    (7700, "results/experiment_7700_v671_record_span_protocol.json", "record_protocol_ready_score"),
    (7701, "results/experiment_7701_v671_sealed_cohort.json", "cohort_ready_score"),
    (7702, "results/experiment_7702_v671_qwen_record_pilot.json", "qwen_pilot_complete_score"),
    (
        7703,
        "results/experiment_7703_v671_typed_decision_energy.json",
        "decision_energy_ready_score",
    ),
    (
        7704,
        "results/experiment_7704_v671_heldout_decisions.json",
        "decision_measurement_complete_score",
    ),
    (
        7705,
        "results/experiment_7705_v671_constraint_bank_protocol.json",
        "constraint_bank_ready_score",
    ),
    (
        7706,
        "results/experiment_7706_v671_continuous_acquisition.json",
        "continuous_acquisition_complete_score",
    ),
    (7707, "results/experiment_7707_v671_independent_evidence_audit.json", None),
    (7708, "results/experiment_7708_v671_arc_generalization_runner.json", None),
    (7709, "results/experiment_7709_v671_arc_first_contact.json", None),
    (7710, "results/experiment_7710_v671_native_record_contract.json", None),
    (7711, "results/experiment_7711_v671_whole_service_cost.json", None),
    (7712, "results/experiment_7712_v671_capstone.json", None),
]
SCOPE = {
    "test_paths": ["tests/python/test_experiment_7707_v671_independent_evidence_audit.py"],
    "changed_modules": [
        "python/carnot/reporting/evidence_audit_v671.py",
        "python/carnot/experiment_7707_v671_independent_evidence_audit.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7707_v671_independent_evidence_audit.py"],
    "requirements": ["REQ-REPORT-7707", "REQ-CL-7707"],
}
GATES = (
    "validity",
    "readiness",
    "coverage",
    "freshness",
    "probability",
    "utility",
    "retention",
    "efficiency",
)


def progress(start: float, phase: str, event: str, **detail: Any) -> None:
    """Expose every phase boundary with elapsed time and completed units."""
    print(
        f"[exp7707] {phase} {event} elapsed_s={time.monotonic() - start:.3f} {detail}", flush=True
    )


def collect(
    root: Path,
) -> tuple[
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Authenticate planned dispositions and cold-reduce the available static rows."""
    custody, hashes, failed = audit_inventory(root, PLAN)
    conductor = root / "results/experiment_7706_continuous_acquisition.json"
    if conductor.is_file():
        hashes["pre_gate_receipts"][str(conductor.relative_to(root))] = sha256_file(conductor)
        custody[7]["conductor_gate_receipt"] = {
            "path": str(conductor.relative_to(root)),
            "sha256": sha256_file(conductor),
            "status": json.loads(conductor.read_bytes()).get("status"),
        }
    path = root / "results/raw/experiment_7704_v671_heldout_decisions/measurement_checkpoint.json"
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"static": None, "online": None}
    if custody[5]["state"] == "producer" and path.is_file():
        hashes["producers"][str(path.relative_to(root))] = sha256_file(path)
        try:
            rows, metrics["static"] = reduce_decisions(json.loads(path.read_bytes())["rows"])
        except (KeyError, TypeError, ValueError) as error:
            failed.append(
                {
                    "check": "raw_row_consistency",
                    "upstream": "Exp7704",
                    "path": str(path.resolve()),
                    "field": "rows",
                    "operator": "==",
                    "expected": "internally_consistent",
                    "observed": str(error),
                }
            )
    elif custody[5]["state"] == "producer":
        failed.append(
            {
                "check": "raw_exists",
                "upstream": "Exp7704",
                "path": str(path.resolve()),
                "field": "exists",
                "operator": "==",
                "expected": True,
                "observed": False,
            }
        )
        hashes["missing_inputs"].append(str(path.relative_to(root)))
    return custody, hashes, failed, rows, metrics


def build_artifact(
    root: Path, date: str, collected: tuple[Any, Any, Any, Any, Any]
) -> dict[str, Any]:
    """Separate administrative completion from absent empirical learning evidence."""
    custody, hashes, failed, rows, metrics = collected
    complete = not failed and metrics["static"] is not None and metrics["online"] is not None
    invalid_rows = any(item["check"] == "raw_row_consistency" for item in failed)
    verdict = (
        "complete_disqualified_inconsistent_evidence"
        if invalid_rows
        else (
            "complete_null_no_registered_benefit"
            if complete
            else "complete_blocked_required_evidence"
        )
    )
    verdict_class = "disqualified" if invalid_rows else ("null" if complete else "blocked")
    static_n = metrics["static"]["independent_families"] if metrics["static"] else None
    operands = {
        "validity": {
            "failed_checks": len(failed),
            "eligible_producers": sum(r["state"] == "producer" for r in custody[:8]),
        },
        "readiness": {
            "required_producers": 7,
            "eligible_producers": sum(r["state"] == "producer" for r in custody[1:8]),
        },
        "coverage": {
            "static_families": static_n,
            "online_update_families": None,
            "online_admission_families": None,
        },
        "freshness": {"static_injected_label_families": static_n, "online_fresh_families": None},
        "probability": {
            "static_by_arm": metrics["static"]["by_arm"] if metrics["static"] else None,
            "online_brier": None,
        },
        "utility": {
            "static_costs": {a: v["cost_mean"] for a, v in metrics["static"]["by_arm"].items()}
            if metrics["static"]
            else None,
            "online_cost": None,
        },
        "retention": {"feedback_events": None, "later_use": None},
        "efficiency": {"online_spend": None, "paired_cost": None},
    }
    principles = {
        "validity": "Invalid evidence must not propagate.",
        "readiness": "Administrative readiness requires eligible static and online evidence.",
        "coverage": "One independent source family counts once across arms and views.",
        "freshness": "Prior exposure and fixture truth cannot confirm natural benefit.",
        "probability": "Quality thresholds require paired proper losses and labels.",
        "utility": "Quality thresholds require measured decisions and costs.",
        "retention": "Retention bounds prevent apparent improvement by forgetting.",
        "efficiency": "Actual spend and later quality must both be measured.",
    }
    gates = {
        name: {
            "passed": complete if name in ("validity", "readiness") else None,
            "measured_operands": operands[name],
            "principle": principles[name],
        }
        for name in GATES
    }
    artifact = {
        "schema": "carnot.exp7707.independent_evidence_audit.v1",
        "experiment_id": 7707,
        "milestone": "2026.09.671",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended": {
                "static_families": 40,
                "online_update_families": 60,
                "online_admission_families": 60,
                "retention_families": 32,
            },
            "observed": {
                "static_families": static_n,
                "online_update_families": None,
                "online_admission_families": None,
                "retention_families": None,
            },
            "eligible": {"static_families": static_n, "online_families": None},
            "excluded": {"static": sum(row["excluded"] for row in rows), "online": None},
            "censored": {"static": sum(row["censored"] for row in rows), "online": None},
            "effective_blocks": {"static": static_n, "online": None},
            "prior_exposure": "Exp7700 fixtures and Exp7702 pilot are exposed; static evaluation uses injected labels.",
            "inference_limits": "Arms and transformed views do not enlarge n; absent online events give no losses.",
        },
        "inference_substrate": "cpu_aggregation",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no_current_model",
        "model_invoked": False,
        "invocation_counts": {
            key: 0
            for key in ("loads", "forwards", "generations", "tokens", "failures", "cancellations")
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "effective_agent_backend": "codex",
        "execution_recovery_receipt": {
            "effective_backend": "codex",
            "force_experiments": os.getenv("CODEX_FORCE_EXPERIMENTS"),
            "successful_current_invocation": bool(os.getenv("CODEX_SESSION_ID")),
            "session_id": os.getenv("CODEX_SESSION_ID"),
            "future_quota_guaranteed": False,
        },
        "phase_spans": [],
        "duration_s": 0.0,
        "random_seed": {"seed": 7707, "purpose": "fixed custody and row order; no resampling"},
        "source_artifact_hashes": hashes,
        "preconditions_checked": {
            "root": str(root.resolve()),
            "host_cpu_available": os.cpu_count() is not None,
            "planned_producer_count": 14,
            "missing_required_count": sum(r["state"] == "missing" for r in custody[1:8]),
        },
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_commands": [],
            "terminal_readers": [],
        },
        "verifier_is_oracle": True,
        "independent_audit_complete_score": int(complete),
        "claim_boundary_matrix": {
            "protocol_validity": "Exp7700 exact fixture protocol only",
            "addressing": "Exp7702 bounded Qwen pilot; no answer truth from a quote",
            "relation_truth": "Exact fixture oracle; circular_positive only",
            "answer_decisions": "Exp7704 injected-error labels and forty independent source families",
            "learning": "unobserved: Exp7706 blocked before online measurement",
            "current_model_provenance": "No model loaded or invoked by Exp7707",
        },
        "recomputed_metrics": metrics,
        "upstream_dispositions": custody,
        "private_control_results": challenge_boundaries(),
        "prior_failures": [
            {
                "experiment_id": "exp7693-independent-evidence-audit",
                "custody": "not_emitted_usage_limit_three_attempts",
            },
            {
                "experiment_id": "exp7679-independent-evidence-audit",
                "custody": "complete_blocked_required_static_online_evidence",
            },
        ],
        "activation": False,
    }
    artifact["field_principles"] = {
        key: (
            "Exact operands and provenance make this field independently checkable."
            if key not in principles
            else principles[key]
        )
        for key in artifact
    } | {key: principles[key] for key in GATES}
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": hashes,
            "config": SCOPE,
            "seed": 7707,
            "reducer_code": sha256_file(root / "python/carnot/reporting/evidence_audit_v671.py"),
        }
    )
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """Reopen immutable source bytes in a fresh process and compare raw reduction."""
    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    custody, hashes, failed, rows, metrics = collect(root)
    errors = []
    for name, observed, expected in (
        ("custody", custody, value["upstream_dispositions"]),
        ("hashes", hashes, value["source_artifact_hashes"]),
        ("gates", failed, value["gate_check_summary"]),
        ("rows", rows, value["rows"]),
        ("metrics", metrics, value["recomputed_metrics"]),
    ):
        if observed != expected:
            errors.append(f"{name}_changed")
    checksum = canonical_hash(
        {
            "sources": hashes,
            "config": SCOPE,
            "seed": 7707,
            "reducer_code": sha256_file(root / "python/carnot/reporting/evidence_audit_v671.py"),
        }
    )
    if checksum != value["reproducibility_checksum"]:
        errors.append("reproducibility_checksum_changed")
    return errors


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Freeze scope, validate, run exact-candidate readers, and publish atomically."""
    root = root.resolve()
    start = time.monotonic()
    spans: list[dict[str, Any]] = []
    last = 0.0

    def span(name: str, units: int) -> None:
        nonlocal last
        now = time.monotonic() - start
        spans.append(
            {
                "phase": name,
                "start_s": last,
                "end_s": now,
                "completed_units": units,
                "heartbeat_times_s": [now],
                "checkpoint": f"{name}:{units}",
            }
        )
        last = now

    progress(start, "preconditions", "start", root=str(root))
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "frozen_affected_scope.json", SCOPE)
    collected = collect(root)
    progress(
        start,
        "preconditions",
        "complete",
        completed_units=len(collected[0]),
        blocked=len(collected[2]),
    )
    span("preconditions", len(collected[0]))
    progress(start, "reduction", "start")
    artifact = build_artifact(root, date, collected)
    atomic_json(
        raw / "checkpoint.json",
        {
            "completed_units": len(artifact["rows"]),
            "source_hashes": artifact["source_artifact_hashes"],
        },
    )
    progress(start, "reduction", "complete", completed_units=len(artifact["rows"]))
    span("reduction", len(artifact["rows"]))
    private = Path(tempfile.mkdtemp(prefix="exp7707-", dir="/tmp"))
    basetemp = private / "basetemp"
    basetemp.mkdir()
    progress(start, "affected_validation", "start")
    commands = checks.build_scoped_commands(
        root,
        SCOPE["test_paths"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=basetemp,
        coverage_file=private / "coverage.data",
    )
    receipts = checks.run_commands(
        root,
        commands,
        log_dir=raw / "validation/affected",
        extra_env={"COVERAGE_FILE": str(private / "coverage.data"), "JAX_PLATFORMS": "cpu"},
    )
    result = checks.reduce_required_checks(receipts)
    artifact["validation_receipts"].update(result)
    artifact["validation_receipts"]["required_commands"] = receipts
    if not result["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
    progress(
        start,
        "affected_validation",
        "complete",
        completed_units=len(receipts),
        passed=result["required_checks_passed"],
    )
    span("affected_validation", len(receipts))
    artifact["phase_spans"] = deepcopy(spans)
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
                "-m",
                "carnot.experiment_7707_v671_independent_evidence_audit",
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
    progress(start, "terminal_readers", "start", candidate=str(candidate))
    terminal_receipts = checks.run_commands(
        root, terminal, log_dir=raw / "validation/terminal", extra_env={"JAX_PLATFORMS": "cpu"}
    )
    artifact["validation_receipts"]["terminal_readers"] = terminal_receipts
    artifact["validation_receipts"]["exact_candidate_sha256"] = sha256_file(candidate)
    artifact["flagged_adversarial"] = not next(
        item["passed"] for item in terminal_receipts if item["name"] == "adversarial_verify"
    )
    if not all(item["passed"] for item in terminal_receipts):
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
    progress(
        start,
        "terminal_readers",
        "complete",
        completed_units=len(terminal_receipts),
        passed=all(item["passed"] for item in terminal_receipts),
    )
    span("terminal_readers", len(terminal_receipts))
    artifact["phase_spans"] = spans
    artifact["duration_s"] = last
    progress(start, "publication", "start", output=str(output))
    atomic_json(output, artifact)
    progress(start, "publication", "complete", completed_units=1)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Expose the run and fresh-process reader through one thin command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", type=Path, default=ROOT / PLAN[8][1])
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    if args.cold:
        errors = cold_replay(args.cold)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    run_experiment(ROOT, args.date, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
