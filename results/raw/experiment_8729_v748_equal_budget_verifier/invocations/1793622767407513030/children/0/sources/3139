"""Publish the V672 independent evidence audit (REQ-REPORT-7721)."""

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
from carnot.reporting.evidence_audit_v672 import failed_check, inspect_sources, reduce_bundle


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7721_v672_independent_evidence_audit")
OUTPUT = Path("results/experiment_7721_v672_independent_evidence_audit.json")
PLAN = {
    7704: "results/experiment_7704_v671_heldout_decisions.json",
    7706: "results/experiment_7706_continuous_acquisition.json",
    7707: "results/experiment_7707_v671_independent_evidence_audit.json",
    7709: "results/experiment_7709_v671_arc_first_contact.json",
    7710: "results/experiment_7710_v671_native_record_contract.json",
    7712: "results/experiment_7712_v671_capstone.json",
    7713: "results/experiment_7713_v672_contract_methods.json",
    7714: "results/experiment_7714_v672_alignment_protocol.json",
    7715: "results/experiment_7715_v672_natural_source_cohort.json",
    7716: "results/experiment_7716_v672_qwen_semantic_pilot.json",
    7717: "results/experiment_7717_latent_evidence_fit.json",
    7718: "results/experiment_7718_v672_natural_decisions.json",
    7719: "results/experiment_7719_v672_acquisition_qualification.json",
    7720: "results/experiment_7720_v672_continuous_acquisition.json",
}
REQUIRED = {7718, 7720}
SCOPE = {
    "test_paths": ["tests/python/test_experiment_7721_v672_independent_evidence_audit.py"],
    "changed_modules": [
        "python/carnot/reporting/evidence_audit_v672.py",
        "python/carnot/experiment_7721_v672_independent_evidence_audit.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7721_v672_independent_evidence_audit.py"],
    "requirements": ["REQ-REPORT-7721", "REQ-CL-7721-AUDIT"],
}
GATES = (
    "measured_validity",
    "readiness",
    "probability",
    "utility",
    "coverage",
    "source_dependence",
    "retention",
    "efficiency",
)
PRINCIPLE = "Measured evidence bounds the claim and prevents invalid downstream use."


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Print a flushed phase boundary with real elapsed time and completed units."""
    print(
        f"[exp7721] {phase} {event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def make_artifact(
    root: Path,
    date: str,
    custody: list[dict[str, Any]],
    hashes: dict[str, Any],
    failures: list[dict[str, Any]],
    reduction: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep source eligibility and claim eligibility separate from completion."""
    states = {row["upstream_id"]: row["state"] for row in custody}
    static = states.get("Exp7718") == "valid" and reduction is not None
    online = states.get("Exp7720") == "valid" and reduction is not None
    eligible = static and online and not failures and not reduction["failed_checks"]
    verdict = (
        "complete_null_no_registered_benefit"
        if eligible
        else "complete_blocked_required_v672_evidence"
    )
    gates = {
        name: {"passed": None, "measured_operands": None, "principle": PRINCIPLE} for name in GATES
    }
    gates["measured_validity"].update(
        passed=not failures, measured_operands={"failed_checks": len(failures)}
    )
    gates["readiness"].update(
        passed=eligible, measured_operands={"static": static, "online": online}
    )
    if reduction is not None:
        gates["coverage"]["measured_operands"] = reduction["sample_size"]
        gates["probability"]["measured_operands"] = reduction["by_arm"]
        gates["source_dependence"]["measured_operands"] = {
            "latent_vs_pooled_brier": reduction["latent_vs_pooled_brier"],
            "frozen_best_arm": reduction["frozen_best_arm"],
        }
    rows = (
        reduction["rows"]
        if reduction
        else [
            {
                "upstream_id": row["upstream_id"],
                "arm": "custody",
                "state": row["state"],
                "raw_metrics": None,
                "denominators": None,
                "excluded": row["state"] != "valid",
                "censored": False,
                "provenance": row["artifact_path"],
            }
            for row in custody
        ]
    )
    sample = reduction["sample_size"] if reduction else None
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7721.v672.independent_evidence_audit.v1",
        "experiment_id": 7721,
        "milestone": "2026.09.672",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": "null" if eligible else "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": rows,
        "per_game_results": [],
        "sample_size_budget": {
            "intended": {
                "static_families": 40,
                "online_update_families": 60,
                "online_admission_families": 60,
                "retention_families": 32,
            },
            "observed": sample,
            "eligible": sample if eligible else None,
            "excluded": None,
            "censored": None,
            "roles": None,
            "exposure": None,
            "effective_blocks": None,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
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
        "effective_execution_backend": "host_cpu_aggregation",
        "phase_spans": [],
        "duration_s": 0.0,
        "random_seed": {"seed": 7721, "purpose": "fixed audit order; no arm selection"},
        "source_artifact_hashes": hashes,
        "preconditions_checked": {
            "root": str(root.resolve()),
            "host_cpu_available": os.cpu_count() is not None,
            "required_static_exists": states.get("Exp7718") != "missing",
            "required_online_exists": states.get("Exp7720") != "missing",
        },
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_commands": [],
            "e2e_checks": [],
            "terminal_readers": [],
            "global_debt": [],
        },
        "verifier_is_oracle": False,
        "independent_audit_complete_score": int(eligible),
        "independent_static_eligible": bool(static),
        "independent_online_eligible": bool(online),
        "claim_matrix": {
            name: {"eligible": False, "operands": None}
            for name in ("probability", "utility", "source_dependence", "acquisition", "retention")
        },
        "static_disposition": "eligible" if static else "blocked",
        "online_disposition": "eligible" if online else "blocked",
        "claim_eligibility_disposition": "eligible_null" if eligible else "blocked",
        "upstream_dispositions": custody,
        "recomputed_metrics": reduction,
        "source_plan": {str(number): path for number, path in PLAN.items()},
        "prior_failures": [
            {
                "experiment_id": "exp7707-independent-evidence-audit",
                "verdict": "complete_blocked_required_evidence",
                "same_verdict_retirement": "V671 required evidence only",
            },
            {
                "experiment_id": "exp7679-independent-evidence-audit",
                "verdict": "complete_blocked_required_static_online_evidence",
                "same_verdict_retirement": "prior static and online evidence only",
            },
        ],
        "activation": False,
    }
    artifact["field_principles"] = {key: PRINCIPLE for key in artifact}
    artifact["field_principles"].update({key: PRINCIPLE for key in GATES})
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "sources": hashes,
            "scope": SCOPE,
            "seed": 7721,
            "reducer_code": sha256_file(root / "python/carnot/reporting/evidence_audit_v672.py")
            if (root / "python/carnot/reporting/evidence_audit_v672.py").is_file()
            else None,
        }
    )
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """Reopen upstream bytes in a fresh process and compare the exact reduction."""
    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    plan = {int(number): path for number, path in value["source_plan"].items()}
    custody, hashes, failures = inspect_sources(root, plan, REQUIRED & set(plan))
    raw_path = root / RAW / "frozen_raw_bundle.json"
    if not failures and REQUIRED <= set(plan):
        if raw_path.is_file():
            reduced = reduce_bundle(json.loads(raw_path.read_bytes()))
            if reduced["failed_checks"]:
                failures.append(
                    failed_check(
                        "raw_bundle_consistency",
                        "current",
                        raw_path,
                        "failed_checks",
                        [],
                        reduced["failed_checks"],
                    )
                )
        else:
            failures.append(
                failed_check(
                    "raw_bundle_exists", "Exp7718+Exp7720", raw_path, "exists", True, False
                )
            )
    errors = []
    for name, current, frozen in (
        ("upstream_dispositions", custody, value["upstream_dispositions"]),
        ("source_artifact_hashes", hashes, value["source_artifact_hashes"]),
        ("gate_check_summary", failures, value["gate_check_summary"]),
    ):
        if current != frozen:
            errors.append(f"{name}_changed")
    if value.get("recomputed_metrics") is not None:
        if not raw_path.is_file():
            errors.append("raw_bundle_missing")
        elif reduce_bundle(json.loads(raw_path.read_bytes())) != value["recomputed_metrics"]:
            errors.append("raw_reduction_changed")
    checksum = canonical_hash(
        {
            "sources": hashes,
            "scope": SCOPE,
            "seed": 7721,
            "reducer_code": sha256_file(root / "python/carnot/reporting/evidence_audit_v672.py")
            if (root / "python/carnot/reporting/evidence_audit_v672.py").is_file()
            else None,
        }
    )
    if checksum != value["reproducibility_checksum"]:
        errors.append("reproducibility_checksum_changed")
    return errors


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Freeze scope, validate, check exact candidate, and publish atomically."""
    root = root.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    last = 0.0

    def span(name: str, units: int) -> None:
        nonlocal last
        now = time.monotonic() - started
        spans.append(
            {
                "phase": name,
                "start_s": last,
                "end_s": now,
                "duration_s": now - last,
                "completed_units": units,
                "heartbeat_timestamps_s": [now],
                "checkpoint": f"{name}:{units}",
            }
        )
        last = now

    progress(started, "preconditions", "start")
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "frozen_affected_scope.json", SCOPE)
    custody, hashes, failures = inspect_sources(root, PLAN, REQUIRED)
    reduction = None
    if not failures:
        raw_bundle = raw / "frozen_raw_bundle.json"
        if raw_bundle.is_file():
            reduction = reduce_bundle(json.loads(raw_bundle.read_bytes()))
            if reduction["failed_checks"]:
                failures.append(
                    failed_check(
                        "raw_bundle_consistency",
                        "current",
                        raw_bundle,
                        "failed_checks",
                        [],
                        reduction["failed_checks"],
                    )
                )
        else:
            failures.append(
                failed_check(
                    "raw_bundle_exists", "Exp7718+Exp7720", raw_bundle, "exists", True, False
                )
            )
    atomic_json(raw / "checkpoint.json", {"completed_units": len(custody), "source_hashes": hashes})
    progress(started, "preconditions", "complete", len(custody))
    span("preconditions", len(custody))
    progress(started, "reduction", "start")
    artifact = make_artifact(root, date, custody, hashes, failures, reduction)
    progress(started, "reduction", "complete", len(artifact["rows"]))
    span("reduction", len(artifact["rows"]))
    private = Path(tempfile.mkdtemp(prefix="exp7721-", dir="/tmp"))
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
    )
    outcome = checks.reduce_required_checks(receipts)
    artifact["validation_receipts"].update(outcome)
    artifact["validation_receipts"]["required_commands"] = receipts
    if not outcome["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
    progress(started, "affected_validation", "complete", len(receipts))
    span("affected_validation", len(receipts))
    progress(started, "repository_health", "start")
    full_suite = checks.run_commands(
        root,
        [
            checks.CommandSpec(
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
                900,
            )
        ],
        log_dir=raw / "validation/full",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=30.0,
    )
    artifact["validation_receipts"]["global_debt"] = full_suite
    progress(started, "repository_health", "complete", len(full_suite))
    span("repository_health", len(full_suite))
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
                "carnot.experiment_7721_v672_independent_evidence_audit",
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
    terminal_receipts = checks.run_commands(
        root, terminal, log_dir=raw / "validation/terminal", extra_env={"JAX_PLATFORMS": "cpu"}
    )
    artifact["validation_receipts"]["terminal_readers"] = terminal_receipts
    artifact["validation_receipts"]["exact_candidate_sha256"] = sha256_file(candidate)
    artifact["flagged_adversarial"] = not next(
        receipt["passed"]
        for receipt in terminal_receipts
        if receipt["name"] == "adversarial_verify"
    )
    if not all(receipt["passed"] for receipt in terminal_receipts):
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["independent_audit_complete_score"] = 0
    progress(started, "terminal_readers", "complete", len(terminal_receipts))
    span("terminal_readers", len(terminal_receipts))
    progress(started, "publication", "start")
    span("publication", 1)
    artifact["phase_spans"] = spans
    artifact["duration_s"] = last
    atomic_json(output, artifact)
    progress(started, "publication", "complete", 1)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Expose the run and independent cold reader through a thin CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
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
