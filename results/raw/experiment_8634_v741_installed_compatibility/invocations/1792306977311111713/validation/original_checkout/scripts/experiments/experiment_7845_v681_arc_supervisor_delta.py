#!/usr/bin/env python3
"""Audit only newly authenticated live supervisor outcomes (REQ-REPORT-7845).

This CLI reads existing bytes and makes a curator recommendation. It never
starts an ARC game, changes a production arm, or imports a numbered runner.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

from carnot.reporting.arc_supervisor_delta import inspect_sources, reduce_rows, verify_prior
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "results/experiment_7831_v680_arc_supervisor_refinement.json"
PRIOR_HASH = "sha256:2d5266fc17b8caa02d19cfe331d8603960c5d72b8aa1a95f8b1b3e5a909d305e"
RAW = ROOT / "results/raw/experiment_7845_v681_arc_supervisor_delta"
MANIFEST = RAW / "validation_command_manifest.json"
DELIVERABLE = ROOT / "results/experiment_7845_v681_arc_supervisor_delta.json"
PRIVATE = Path("/tmp/carnot-exp7845")


def progress(started: float, phase: str, completed: int) -> None:
    """Show progress through each bounded phase so a stalled child stays visible."""

    print(
        f"[exp7845] phase={phase} elapsed_s={time.monotonic() - started:.3f} completed_units={completed}",
        flush=True,
    )


def preconditions() -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Check actual baseline and registry bytes before consuming outcome data."""

    failures = verify_prior(PRIOR, PRIOR_HASH)
    checks: list[dict[str, Any]] = [
        {
            "upstream_id": "exp7831-arc-supervisor-refinement",
            "path": str(PRIOR),
            "sha256": sha256_file(PRIOR) if PRIOR.is_file() else None,
            "artifact_field": "inventory_hash",
            "op": "==",
            "expected": PRIOR_HASH,
            "observed": sha256_file(PRIOR) if PRIOR.is_file() else "missing",
            "role": "historical_dedup_baseline_only",
            "eligibility": "not_science_evidence",
        }
    ]
    hashes: set[str] = set()
    if not failures:
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        for item in prior.get("source_inventory", []):
            label = item.get("raw") or item.get("producer")
            expected = item.get("raw_sha256")
            if not label or not expected:
                continue
            path = ROOT / label
            observed = sha256_file(path) if path.is_file() else "missing"
            check = {
                "upstream_id": "exp7831-arc-supervisor-refinement",
                "path": str(path),
                "sha256": observed if observed != "missing" else None,
                "artifact_field": "raw_sha256",
                "op": "==",
                "expected": expected,
                "observed": observed,
                "role": item.get("role"),
                "eligibility": item.get("disposition"),
            }
            checks.append(check)
            if observed != expected:
                failures.append(check)
            hashes.add(expected)
    registry = ROOT / "ops/arc_solve_registry.yaml"
    observed_registry = sha256_file(registry) if registry.is_file() else "missing"
    registry_text = registry.read_text(encoding="utf-8") if registry.is_file() else ""
    levels_present = bool(re.search(r"^\s*levels_reproduced:\s*[1-9]\d*\s*$", registry_text, re.M))
    registry_check = {
        "upstream_id": "arc_solve_registry",
        "path": str(registry),
        "sha256": observed_registry if observed_registry != "missing" else None,
        "artifact_field": "levels_reproduced_positive",
        "op": "==",
        "expected": True,
        "observed": levels_present,
        "role": "registry_precheck",
        "eligibility": "read_only",
    }
    checks.append(registry_check)
    if not levels_present:
        failures.append(registry_check)
    return checks, failures, hashes


def base_artifact(
    date: str,
    started_ns: int,
    checks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    reduced: dict[str, Any],
) -> dict[str, Any]:
    """Keep per-unit evidence and unmeasured gates explicit in the candidate."""

    counts = {
        state: sum(row.get("status") == state for row in rows)
        for state in ("completed", "censored", "excluded", "shadow", "source")
    }
    count_closed = counts["completed"]
    independent = len(
        {(r.get("game"), r.get("seed")) for r in rows if r.get("status") == "completed"}
    )
    receipt = build_current_work_receipt(
        run_id=f"exp7845-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"source_count": reduced["new_source_count"]},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    source_hashes = {item["path"]: item["sha256"] for item in checks if item.get("sha256")}
    source_hashes.update(
        {r["source_path"]: r["source_sha256"] for r in rows if r.get("source_sha256")}
    )
    for path in (
        ROOT / "python/carnot/reporting/arc_supervisor_delta.py",
        ROOT / "scripts/experiments/experiment_7845_v681_arc_supervisor_delta.py",
        MANIFEST,
    ):
        if path.is_file():
            source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    artifact = {
        "experiment_id": 7845,
        "task_id": "exp7845-arc-supervisor-delta",
        "milestone": "2026.09.681",
        "run_date": date,
        **reduced,
        "gate_check_summary": failures,
        "preconditions_checked": checks,
        "rows": rows
        or [
            {
                "unit": "inventory_delta",
                "source_path": "results/raw",
                "status": "completed",
                "intended": 0,
                "eligible": 0,
                "started": 0,
                "completed": 0,
                "censored": 0,
                "excluded": 0,
                "independent_n": 0,
            }
        ],
        "sample_size_budget": {
            "intended": len(rows),
            "eligible": counts["source"],
            "started": counts["completed"] + counts["censored"],
            "completed": count_closed,
            "censored": counts["censored"],
            "excluded": counts["excluded"],
            "independent_n": independent,
        },
        "acceptance_gate_results": {
            "validity": None,
            "readiness": None,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"calls": 0, "tokens": 0, "file_hashes": []},
        "source_artifact_hashes": source_hashes,
        "random_seed": 0,
        "current_work_receipt": receipt,
        "reproducibility_checksum": canonical_hash(
            {"sources": source_hashes, "seed": 0, "config": sha256_file(MANIFEST)}
        ),
        "verifier_is_oracle": False,
        "claim_scope": "observational live supervisor audit; no causal benefit or new solve",
        "solve_provenance": "live_agent_self_discovery" if reduced["new_source_count"] else None,
        "solve_claim": False,
        "production_defaults_changed": False,
        "supervisor_delta_ready_score": None,
        "validation_command_manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "validation_receipts": [],
        "observed_child_commands": [],
        "repository_health": {
            "status": "unmeasured",
            "historical_full_suite_obligations_open": True,
        },
        "flagged_adversarial": None,
        "phase_spans": [],
        "duration_s": 0.0,
    }
    artifact["field_principles"] = {
        field: "Exact current bytes and bounded claims; unmeasured gates stay null."
        for field in artifact
    }
    return artifact


def cold_replay(candidate: Path) -> list[str]:
    """Reopen sealed child bytes in a fresh CLI process before any final claim."""

    value = json.loads(candidate.read_text(encoding="utf-8"))
    errors: list[str] = []
    for item in value.get("validation_receipts", []):
        path = ROOT / item["log_path"]
        if not path.is_file() or sha256_file(path) != item["log_sha256"]:
            errors.append(item["name"])
    return errors


def seal(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy closed child logs into immutable paths named by their exact bytes."""

    sealed = RAW / "sealed_logs"
    sealed.mkdir(parents=True, exist_ok=True)
    for receipt in receipts:
        source = Path(receipt["log_path"])
        digest = sha256_file(source).removeprefix("sha256:")
        target = sealed / f"{receipt['name']}_{digest}.log"
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("sealed_log_collision")
        if not target.exists():
            shutil.copyfile(source, target)
        receipt["log_path"] = target.relative_to(ROOT).as_posix()
        receipt["log_sha256"] = "sha256:" + digest
    return receipts


def validate(artifact: dict[str, Any], started: float) -> dict[str, Any]:
    """Run exactly the frozen commands; keep health separate from required checks."""

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    commands = manifest["commands"]
    receipts: list[dict[str, Any]] = []
    PRIVATE.mkdir(parents=True, exist_ok=True)
    previous = json.loads(DELIVERABLE.read_text(encoding="utf-8")) if DELIVERABLE.is_file() else {}
    old_health = next(
        (
            row
            for row in previous.get("validation_receipts", [])
            if row.get("name") == "repository_health_180s"
        ),
        None,
    )
    for index, item in enumerate(commands):
        if item["name"] == "cold_replay":
            candidate = PRIVATE / "candidate.json"
            artifact["validation_receipts"] = list(receipts)
            artifact["duration_s"] = time.monotonic() - started
            artifact["phase_spans"] = [{"phase": "candidate", "duration_s": artifact["duration_s"]}]
            atomic_json(candidate, artifact)
        if item["name"] == "repository_health_180s" and old_health is not None:
            old_log = ROOT / old_health["log_path"]
            if (
                old_health.get("command_argv") == item["argv"]
                and old_log.is_file()
                and sha256_file(old_log) == old_health.get("log_sha256")
            ):
                progress(started, "before_reuse_repository_health_180s", index)
                reused = dict(old_health, reused_from_sha256=sha256_file(DELIVERABLE))
                receipts.append(reused)
                progress(started, "after_reuse_repository_health_180s", index + 1)
                continue
        spec = CommandSpec(
            item["name"], tuple(item["argv"]), item["classification"], float(item["deadline_s"])
        )
        progress(started, f"before_{spec.name}", index)
        outcome = run_commands(
            ROOT, [spec], log_dir=PRIVATE / "logs" / str(index), heartbeat_s=30.0
        )
        outcome = seal(outcome)
        outcome[0]["class"] = item["classification"]
        receipts.extend(outcome)
        progress(started, f"after_{spec.name}", index + 1)
    by_name = {row["name"]: row for row in receipts}
    required = [row["name"] for row in commands if row["classification"] == "required"]
    failures = [name for name in required if by_name.get(name, {}).get("passed") is not True]
    imports = by_name["worktree_imports"].get("resolved_imports")
    if (
        not isinstance(imports, dict)
        or not imports
        or any(not Path(path).is_relative_to(ROOT / "python") for path in imports.values())
    ):
        failures.append("worktree_imports_resolved_paths")
    artifact["validation_receipts"] = receipts
    artifact["observed_child_commands"] = [
        row["command_argv"] for row in receipts if "reused_from_sha256" not in row
    ]
    artifact["validation_errors"] = failures
    health = by_name["repository_health_180s"]
    artifact["repository_health"] = {
        "status": "healthy" if health["passed"] else "failed_health",
        "diagnostic": health,
        "historical_full_suite_obligations_open": True,
        "historical_failure": {
            "path": PRIOR.relative_to(ROOT).as_posix(),
            "required_check": "all_python_tests",
            "exit_code": -15,
            "verdict": "complete_disqualified_required_validation",
        },
    }
    artifact["flagged_adversarial"] = by_name["adversarial_verify"]["exit_code"] != 0
    valid = (
        not failures and not artifact["gate_check_summary"] and not artifact["flagged_adversarial"]
    )
    artifact["acceptance_gate_results"]["validity"] = valid
    artifact["acceptance_gate_results"]["readiness"] = int(valid)
    artifact["supervisor_delta_ready_score"] = int(valid)
    if artifact["gate_check_summary"]:
        artifact["honest_verdict"] = "complete_blocked_source_precondition"
        artifact["verdict_class"] = "blocked"
    elif failures or artifact["flagged_adversarial"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    return artifact


def main() -> int:
    """Run the read-only experiment or inspect immutable child logs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--reader-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    progress(started, "start", 0)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    progress(started, "preconditions_before", 0)
    checks, failures, hashes = preconditions()
    progress(started, "preconditions_after", len(checks))
    rows: list[dict[str, Any]] = []
    if not failures:
        progress(started, "scan_before", 0)
        rows = inspect_sources(ROOT, "20260928", hashes)
        progress(started, "scan_after", len(rows))
    reduced = reduce_rows(rows)
    artifact = base_artifact(args.date, started_ns, checks, failures, rows, reduced)
    if args.reader_only:
        artifact["honest_verdict"] = "partial_private_reader_e2e"
        artifact["verdict_class"] = "partial"
        output = args.output or PRIVATE / "e2e/result.json"
        artifact["duration_s"] = time.monotonic() - started
        atomic_json(output, artifact)
        progress(started, "private_output_written", len(rows))
        return 0
    progress(started, "validation_before", 0)
    artifact = validate(artifact, started)
    artifact["duration_s"] = time.monotonic() - started
    artifact["phase_spans"] = [{"phase": "total", "duration_s": artifact["duration_s"]}]
    artifact["field_principles"].update(
        {
            key: "Measured current result; historical failures remain separate."
            for key in ("validation_receipts", "repository_health", "duration_s")
        }
    )
    atomic_json(args.output or DELIVERABLE, artifact)
    progress(started, "deliverable_written", len(rows))
    return 0 if artifact["supervisor_delta_ready_score"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
