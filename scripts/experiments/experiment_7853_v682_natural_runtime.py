#!/usr/bin/env python3
"""Exp7853 preflight and terminal custody for REQ-VERIFY-7853."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.verify import natural_bank, natural_predicates, natural_training

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7853_v682_natural_runtime.json"
SOURCES = (
    (
        "exp7852-source-boundary",
        "results/experiment_7852_v682_source_boundary.json",
        "producer",
        "carnot.exp7852.source_boundary.v1",
    ),
    (
        "exp7852-public",
        "results/raw/experiment_7852_v682_source_boundary/public_manifest.json",
        "public_manifest",
        "carnot.exp7852.public.v1",
    ),
    (
        "exp7852-evaluator",
        "results/raw/experiment_7852_v682_source_boundary/evaluator_manifest.json",
        "private_manifest",
        "carnot.exp7852.evaluator.v1",
    ),
    (
        "exp7825-fixture",
        "results/experiment_7825_v680_training_runtime.json",
        "historical_fixture",
        None,
    ),
)
SEEDS = (67801, 67802, 67803)
MODEL_SPECS: list[dict[str, Any]] = []


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Keep long owned steps observable without inventing elapsed compute."""
    print(
        f"[exp7853] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def _check(
    upstream: str, path: Path, digest: str | None, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    return {
        "upstream_id": upstream,
        "path": str(path),
        "sha256": digest,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check source identity, schema and readiness before any numerical work."""
    checks: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    values: dict[str, dict[str, Any]] = {}
    for upstream, relative, role, schema in SOURCES:
        path = root / relative
        exists = path.is_file()
        digest = sha256_file(path) if exists else None
        sources.append(
            {
                "upstream_id": upstream,
                "path": str(path),
                "sha256": digest,
                "role": role,
                "exposure": "historical" if role == "historical_fixture" else "exposed_development",
            }
        )
        checks.append(_check(upstream, path, digest, "exists", True, exists))
        if exists:
            try:
                values[upstream] = json.loads(path.read_text())
            except (ValueError, OSError):
                values[upstream] = {}
            if schema:
                checks.append(
                    _check(upstream, path, digest, "schema", schema, values[upstream].get("schema"))
                )
    producer = values.get("exp7852-source-boundary", {})
    path = root / SOURCES[0][1]
    digest = sources[0]["sha256"]
    for field, expected in (
        ("experiment_id", 7852),
        ("milestone", "2026.09.682"),
        ("run_date", "20260929"),
        ("source_boundary_ready_score", 1),
    ):
        checks.append(
            _check("exp7852-source-boundary", path, digest, field, expected, producer.get(field))
        )
    checks.append(
        _check(
            "exp7852-source-boundary",
            path,
            digest,
            "verdict_class",
            "positive",
            producer.get("verdict_class"),
        )
    )
    for manifest_id, hash_field in (
        ("exp7852-public", "public_manifest_sha256"),
        ("exp7852-evaluator", "evaluator_manifest_sha256"),
    ):
        manifest = next(row for row in sources if row["upstream_id"] == manifest_id)
        checks.append(
            _check(
                "exp7852-source-boundary",
                path,
                digest,
                hash_field,
                manifest["sha256"],
                producer.get(hash_field),
            )
        )
    return checks, sources


def blocked_record(date: str, root: Path = ROOT) -> dict[str, Any]:
    """Retain every failed upstream operand; fixture readiness has no authority."""
    started = time.monotonic_ns()
    checks, sources = preflight(root)
    failures = [
        {key: value for key, value in row.items() if key != "passed"}
        for row in checks
        if not row["passed"]
    ]
    if not failures:
        raise RuntimeError("qualified source path requires the natural measurement runner")
    ended = time.monotonic_ns()
    spans = [{"phase": "preflight", "duration_s": (ended - started) / 1e9, "completed_units": 0}]
    receipt = build_current_work_receipt(
        run_id=f"exp7853-{date}-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_no_pretrained_model",
        inference_substrate_details={"model_files": []},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=spans,
    )
    rows = [
        {
            "arm": arm,
            "seed": seed,
            "source_family": None,
            "status": "unstarted_external_precondition",
            "intended": 1,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "censored": 0,
            "excluded": 0,
            "independent": 0,
        }
        for arm in natural_training.ARMS
        for seed in SEEDS
    ]
    gates = {
        name: (False if name in {"validity", "readiness"} else None)
        for name in (
            "validity",
            "readiness",
            "probability_quality",
            "decision_benefit",
            "retention",
            "efficiency",
        )
    }
    return {
        "schema": "carnot.exp7853.natural_runtime.v1",
        "experiment_id": 7853,
        "task_id": "exp7853-natural-runtime",
        "milestone": "2026.09.682",
        "run_date": date,
        "honest_verdict": "complete_blocked_disqualified_source_boundary",
        "verdict_class": "blocked",
        "flagged_adversarial": None,
        "gate_check_summary": failures,
        "rows": rows,
        "sample_size_budget": {
            "intended": len(rows),
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "censored": 0,
            "excluded": 0,
            "independent": 0,
        },
        "acceptance_gate_results": gates,
        "duration_s": (ended - started) / 1e9,
        "phase_spans": spans,
        "random_seed": list(SEEDS),
        "source_artifact_hashes": sources,
        "preconditions_checked": checks,
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "repository_health": {"status": "not_run"},
        "verifier_is_oracle": False,
        "claim_scope": "blocked_before_natural_measurement",
        "inference_substrate": "cpu_no_pretrained_model",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "model_file_hashes": []},
        "natural_training_ready_score": 0,
        "natural_online_ready_score": 0,
        "predicate_protocol_path": str(root / "python/carnot/verify/natural_predicates.py"),
        "training_protocol_path": str(root / "python/carnot/verify/natural_training.py"),
        "online_protocol_path": str(root / "python/carnot/verify/natural_bank.py"),
        "fixture_rows": [],
        "current_work_receipt": receipt,
        "resolved_imports": {
            "natural_predicates": str(Path(natural_predicates.__file__).resolve()),
            "natural_training": str(Path(natural_training.__file__).resolve()),
            "natural_bank": str(Path(natural_bank.__file__).resolve()),
        },
        "historical_obligations": {
            "exp7825": "fixture readiness only; prior required failures retained in source artifact",
            "full_python_suite": "unverified historical requirement",
        },
    }


def finalize_record(record: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Bind current code and upstream bytes without hashing the output itself."""
    code = [
        root / "python/carnot/verify/natural_predicates.py",
        root / "python/carnot/verify/natural_training.py",
        root / "python/carnot/verify/natural_bank.py",
        root / "scripts/experiments/experiment_7853_v682_natural_runtime.py",
    ]
    record["code_hashes"] = [{"path": str(path), "sha256": sha256_file(path)} for path in code]
    record["reproducibility_checksum"] = canonical_hash(
        {
            "code": record["code_hashes"],
            "sources": record["source_artifact_hashes"],
            "seeds": record["random_seed"],
            "schema": record["schema"],
        }
    )
    principles = {
        "experiment_id": "Identify the current science producer.",
        "task_id": "Avoid treating legacy aliases as this task.",
        "honest_verdict": "External disqualification terminates before measurement.",
        "verdict_class": "Let downstream gates distinguish blocked and failed work.",
        "flagged_adversarial": "A terminal verifier result must control trust.",
        "gate_check_summary": "Name every failed upstream operand exactly.",
        "rows": "Keep every intended arm and seed visible despite the block.",
        "sample_size_budget": "Do not count views or seeds as independent source groups.",
        "acceptance_gate_results": "Do not turn unmeasured benefit into success.",
        "duration_s": "Report measured owner time.",
        "phase_spans": "Show which work completed before the block.",
        "random_seed": "Freeze all intended registered seeds.",
        "reproducibility_checksum": "Bind code, input, configuration and seeds.",
        "source_artifact_hashes": "Keep historical and current authority distinct.",
        "preconditions_checked": "Preserve both passing and failed gate operands.",
        "validation_receipts": "Required failures cannot be erased by a wrapper.",
        "validation_command_manifest_path": "Make validation argv reviewable.",
        "observed_child_commands": "Attribute only owned child processes.",
        "repository_health": "A diagnostic is separate from science gates.",
        "verifier_is_oracle": "Fixtures do not certify natural truth.",
        "claim_scope": "State why natural evidence is absent.",
        "inference_substrate": "Use the duration class of actual CPU work.",
        "inference_substrate_class": "No model was loaded.",
        "MODEL_SPECS": "No pretrained generator belongs to this task.",
        "model_specs": "Do not inherit historical model calls.",
        "model_invocation_counts": "Only count current model activity.",
        "natural_training_ready_score": "The source gate must open before natural training.",
        "natural_online_ready_score": "The source gate must open before online learning.",
        "predicate_protocol_path": "Freeze the public-byte feature grammar.",
        "training_protocol_path": "Freeze the disjoint numerical adapter.",
        "online_protocol_path": "Freeze delayed feedback and admission rules.",
        "fixture_rows": "Fixture checks do not replace natural evidence.",
    }
    record["field_principles"] = {
        key: principles.get(key, "Preserve this field for exact replay and provenance.")
        for key in record
    }
    for gate in record["acceptance_gate_results"]:
        record["field_principles"][f"acceptance_gate_results.{gate}"] = (
            "Keep this gate separate so unmeasured claims remain null."
        )
    return record


def cold_replay(path: Path) -> dict[str, Any]:
    """Reopen exact upstream bytes and reject candidate drift in another process."""
    record = json.loads(path.read_text())
    checks, sources = preflight(ROOT)
    valid = (
        record["source_artifact_hashes"] == sources
        and record["preconditions_checked"] == checks
        and record["gate_check_summary"]
        == [
            {key: value for key, value in row.items() if key != "passed"}
            for row in checks
            if not row["passed"]
        ]
    )
    code = record["code_hashes"]
    valid = valid and all(sha256_file(Path(row["path"])) == row["sha256"] for row in code)
    checksum = canonical_hash(
        {
            "code": code,
            "sources": sources,
            "seeds": record["random_seed"],
            "schema": record["schema"],
        }
    )
    valid = valid and checksum == record["reproducibility_checksum"]
    return {
        "valid": valid,
        "row_count": len(record["rows"]),
        "failed_operands": len(record["gate_check_summary"]),
    }


def main(argv: list[str] | None = None) -> int:
    start = time.monotonic()
    progress(start, "start", "entered")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        progress(start, "cold_replay", "before")
        result = cold_replay(args.cold_replay)
        print(json.dumps(result, sort_keys=True), flush=True)
        progress(start, "cold_replay", "after", result["row_count"])
        return 0 if result["valid"] else 1
    if args.date != "20260929":
        parser.error("Exp7853 date is frozen to 20260929")
    private = args.private_root or Path(tempfile.mkdtemp(prefix="exp7853-", dir="/tmp"))
    private.mkdir(parents=True, exist_ok=True)
    progress(start, "preflight", "before")
    record = finalize_record(blocked_record(args.date))
    progress(start, "preflight", "after")
    output = args.output or OUTPUT
    atomic_json(output, record)
    progress(start, "publish", "after")
    print(
        json.dumps({"output": str(output), "honest_verdict": record["honest_verdict"]}), flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
