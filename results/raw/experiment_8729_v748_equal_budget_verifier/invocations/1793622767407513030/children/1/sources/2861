"""Build the isolated V644 executor fixture and fresh request panels.

The producer launches a public learner and a separately implemented evaluator.
It reduces their raw receipts after both processes exit. This is an audited
process boundary, not a security sandbox or a live learning efficacy result.

Spec refs: REQ-CL-7330 and SCENARIO-CL-7330-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

from carnot.experiment_7330_v644_public_learner import (
    canonical_bytes,
    sha256_file,
    sha256_json,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260916"
SCHEMA = "carnot.experiment_7330.v644_executor_isolation.v1"
RESULT_NAME = "experiment_7330_v644_executor_isolation.json"
RAW_NAME = "experiment_7330_v644_executor_isolation"
COHORTS = (
    "stable_rules",
    "announced_changes",
    "return_to_prior_version",
    "unannounced_changes",
)
REQUIRED_SOURCE_PATHS = (
    "CLAUDE.md",
    "CODEX.md",
    "research-program.md",
    "research-references.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/continuous-learning/spec.md",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/experiment_7323_v643_addition_prototype.py",
    "python/carnot/experiment_7325_v643_addition_audit.py",
    "python/carnot/memory/transactional_constraint_memory.py",
    "python/carnot/experiment_7330_v644_public_learner.py",
    "python/carnot/experiment_7330_v644_executor_isolation.py",
    "scripts/experiments/experiment_7330_v644_private_executor.py",
    "scripts/experiments/experiment_7330_v644_executor_isolation.py",
    "tests/python/test_experiment_7330_v644_executor_isolation.py",
    "results/experiment_7325_v643_addition_audit.json",
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw evidence, candidates, and the terminal result at fixed paths."""

    output_root: Path
    raw_root: Path
    public_manifest: Path
    private_manifest: Path
    learner_evidence: Path
    evaluator_receipt: Path
    executor_controls: Path
    candidate: Path
    result: Path

    @classmethod
    def for_output_root(cls, output_root: Path) -> ExperimentPaths:
        root = output_root.resolve()
        raw = root / "results/raw" / RAW_NAME
        return cls(
            output_root=root,
            raw_root=raw,
            public_manifest=raw / "public_manifest.json",
            private_manifest=raw / "evaluator/evaluator_private_manifest.json",
            learner_evidence=raw / "learner/learner_evidence.json",
            evaluator_receipt=raw / "evaluator/evaluator_boundary_receipt.json",
            executor_controls=raw / "evaluator/executor_controls.json",
            candidate=raw / "terminal_candidate.json",
            result=root / "results" / RESULT_NAME,
        )


def _load_object(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON with fsync and one same-directory rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - interrupted rename cleanup.
            temporary.unlink()


def _environment(root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    return environment


def _run_streamed(
    argv: Sequence[str], root: Path, log_path: Path, *, timeout_s: float = 120.0
) -> JsonDict:
    """Stream a bounded child and retain its real output and exit status."""

    started = time.monotonic()
    print(f"[exp7330] before_subprocess command={shlex.join(argv)}", flush=True)
    process = subprocess.Popen(  # noqa: S603 - fixed argument vectors, never a shell.
        tuple(argv),
        cwd=root,
        env=_environment(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(f"[exp7330-child] {line.rstrip()}", flush=True)
        if time.monotonic() - started > timeout_s:  # pragma: no cover - bounded kill guard.
            process.terminate()
            break
    exit_code = process.wait(timeout=5)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(lines), encoding="utf-8")
    receipt = {
        "command": shlex.join(argv),
        "exit_code": exit_code,
        "duration_s": time.monotonic() - started,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "output_lines": lines,
        "passed": exit_code == 0,
    }
    print(
        f"[exp7330] after_subprocess exit={exit_code} elapsed_s={receipt['duration_s']:.3f}",
        flush=True,
    )
    if exit_code != 0:
        raise RuntimeError(f"child_failed:{shlex.join(argv)}")
    return receipt


def _tee_reader(
    process: subprocess.Popen[str], prefix: str, lines: list[str], events: queue.Queue[str]
) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(f"[{prefix}] {line.rstrip()}", flush=True)
        events.put(line)


def collect_preconditions(
    repo_root: Path, *, overrides: Mapping[str, Any] | None = None
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate current inputs while keeping V643 evidence diagnostic-only."""

    root = repo_root.resolve()
    hashes = {
        relative: sha256_file(root / relative)
        for relative in REQUIRED_SOURCE_PATHS
        if (root / relative).is_file()
    }
    spec = (root / "openspec/capabilities/continuous-learning/spec.md").read_text(encoding="utf-8")
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    historical = _load_object(root / "results/experiment_7325_v643_addition_audit.json")
    observed: JsonDict = {
        "required_sources_available": len(hashes),
        "capability_requirement_present": "REQ-CL-7330" in spec,
        "scenario_contract_present": spec.count("SCENARIO-CL-7330-") >= 5,
        "current_task_not_quarantined": "experiment_7330_v644_executor_isolation" not in exclusion,
        "no_same_milestone_inputs": [],
        "historical_diagnostic_available": historical.get("status") == "complete",
        "historical_diagnostic_not_readiness_gate": True,
        "host_execution_venue": os.name,
    }
    observed.update(dict(overrides or {}))
    metadata = {
        "required_sources_available": ("repository_worktree", "paths", len(REQUIRED_SOURCE_PATHS)),
        "capability_requirement_present": (
            "openspec/capabilities/continuous-learning/spec.md",
            "REQ-CL-7330",
            True,
        ),
        "scenario_contract_present": (
            "openspec/capabilities/continuous-learning/spec.md",
            "SCENARIO-CL-7330-*",
            True,
        ),
        "current_task_not_quarantined": (
            "ops/exclusion_manifest.yaml",
            "experiment_7330",
            True,
        ),
        "no_same_milestone_inputs": ("milestone_2026.09.644", "inputs", []),
        "historical_diagnostic_available": (
            "results/experiment_7325_v643_addition_audit.json",
            "status",
            True,
        ),
        "historical_diagnostic_not_readiness_gate": (
            "results/experiment_7325_v643_addition_audit.json",
            "readiness_authority",
            True,
        ),
        "host_execution_venue": ("host", "os.name", "posix"),
    }
    checks = []
    for name, value in observed.items():
        upstream, field, expected = metadata[name]
        checks.append(
            {
                "upstream": upstream,
                "check": name,
                "field": field,
                "expected_value": expected,
                "observed_value": value,
                "passed": value == expected,
                "principle": "Input identity and eligibility must be explicit before work starts.",
            }
        )
    return checks, {"current_sources": hashes}


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every check and the first exact failure without paraphrase."""

    failures = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failures,
        "check_count": len(checks),
        "failed_check_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "checks": [deepcopy(dict(row)) for row in checks],
    }


def _field_principles() -> JsonDict:
    return {
        "schema": "Version the artifact while preserving ordinary experiment identity.",
        "status": "Publish a terminal result only after current work and affected validation.",
        "run_date": "Use the declared date and retain actual UTC timestamps.",
        "preconditions_checked": "Name every input identity, availability check, and failure.",
        "MODEL_SPECS": "List only executable identities intended for current model work.",
        "model_invoked": "Mark every real model load or generation attempt.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled, and active work.",
        "inference_substrate": "Declare the computation that produced current evidence.",
        "inference_substrate_class": "Declare the matching duration class.",
        "execution_venue": "Separate host work from historical board evidence.",
        "duration_s": "Measure monotonic elapsed time without padding.",
        "phase_spans": "Explain cost with disjoint spans and completed units.",
        "random_seed": "Freeze public, token, private-rule, and resampling identities.",
        "reproducibility_checksum": "Bind code, settings, manifests, evaluator, and raw evidence.",
        "source_artifact_hashes": "Authenticate exact sources and task-owned sidecars.",
        "rows": "Retain every executed development request with costs and disposition.",
        "sample_size_budget": "Separate planned, attempted, completed, and censored units.",
        "acceptance_gate_results": "Keep expected, observed, and pass state for every gate.",
        "gate_check_summary": "Preserve upstream, field, expected value, and observed value.",
        "verifier_is_oracle": "Executor-defined correctness remains circular authority.",
        "honest_verdict": "Use a terminal prefix and state the evidence boundary.",
        "verdict_class": "Use the closed terminal evidence enum.",
        "validation_receipts": "Retain command, scope, exit, duration, and exact log hash.",
        "repository_health": "Keep unrelated failures distinct from affected checks.",
        "field_principles": "Explain required evidence fields without wrapping their values.",
        "executor_fixture_ready_score": "Require separation, independent checks, seals, invalid rejection, and witnesses.",
        "public_manifest": "Bind public requests and opaque versions apart from private rules.",
        "executor_boundary_receipt": "Record allowed inputs, process IDs, imports, files, and leak controls.",
        "acceptance_contract": "Freeze downstream gates and sample sizes before held-out labels.",
    }


def _acceptance_contract() -> JsonDict:
    return {
        "sealed_before_held_out_output_observed": True,
        "development_streams": 4,
        "held_out_streams": 16,
        "requests_per_stream": 12,
        "warmup_requests_per_stream": 4,
        "later_requests_per_stream": 8,
        "held_out_streams_per_cohort": 4,
        "live_proposal_pairs": 24,
        "live_pairs_per_cohort": 6,
        "query_budget_per_request": 24,
        "downstream_arms": [
            "persistent_acquisition",
            "reset_per_request",
            "exact_plan_cache_plus_reset",
            "frozen_after_warmup",
        ],
        "same_distribution_and_budgets_across_arms": True,
        "runtime_candidate_reads_held_out_labels": False,
        "stopping_rule": "execute each fixed unit once; do not extend from outcomes",
    }


def _repository_health() -> JsonDict:
    return {
        "status": "degraded_open",
        "affects_required_checks": False,
        "historical_failures": [
            {
                "source_experiment_id": 7312,
                "date": "2026-09-14",
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 2,
                "resolved": False,
                "classification": "unrelated_repository_wide_collection_failures",
            }
        ],
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind scientific inputs and raw identities while excluding host timing."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "inference_substrate",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "public_manifest",
        "executor_boundary_receipt",
        "executor_controls",
        "compound_conflict_challenge",
        "acceptance_contract",
        "acceptance_gate_results",
        "verifier_is_oracle",
        "executor_fixture_ready_score",
        "verdict_class",
    )
    return sha256_json({key: artifact.get(key) for key in keys})


def _base_artifact(checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": 7330,
        "milestone": "2026.09.644",
        "phase": 1,
        "status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "loads": {"attempted": 0, "completed": 0, "failed": 0, "cancelled": 0, "in_flight": 0},
            "generations": {
                "attempted": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "in_flight": 0,
            },
        },
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "public_manifest": 7_330_101,
            "opaque_tokens_commitment_only": "sha256-bound evaluator-only seed",
            "private_rules_commitment_only": "sha256-bound evaluator-only seed",
            "resampling": 7_330_401,
        },
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "rows": [],
        "sample_size_budget": {},
        "acceptance_gate_results": {},
        "gate_check_summary": gate_check_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "complete_circular_positive_executor_fixture_ready_no_efficacy_claim",
        "verdict_class": "circular_positive",
        "validation_receipts": [],
        "required_checks_passed": False,
        "missing_required_commands": list(REQUIRED_CHECK_NAMES),
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": _repository_health(),
        "field_principles": _field_principles(),
        "executor_fixture_ready_score": 0,
        "executor_value_score": 0,
        "promotion_score": 0,
        "public_manifest": {},
        "executor_boundary_receipt": {},
        "acceptance_contract": _acceptance_contract(),
        "executor_controls": {},
        "compound_conflict_challenge": {},
        "no_model_weight_mutation": True,
        "production_default_changed": False,
        "publication_surface_changed": False,
        "synthetic_domain_development_evidence": True,
        "satellite_operations_claim": False,
        "live_model_efficacy_claim": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]
) -> JsonDict:
    """Return a row-free terminal block with the canonical failed check."""

    artifact = _base_artifact(checks, hashes)
    failure = gate_check_summary(checks)["first_failure"]
    artifact.update(
        {
            "status": "blocked",
            "honest_verdict": f"blocked_{failure['check']}: prerequisite failed",
            "verdict_class": "blocked",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "sample_size_budget": {
                "planned_units": 48,
                "attempted_units": 0,
                "completed_units": 0,
                "censored_units": 0,
                "fixed_stopping_rule": True,
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _manifest_errors(public_manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    development = public_manifest.get("development_streams", [])
    held_out = public_manifest.get("held_out_streams", [])
    live = public_manifest.get("live_proposal_panel", [])
    if len(development) != 4:
        errors.append("development_stream_count")
    if len(held_out) != 16:
        errors.append("held_out_stream_count")
    if any(len(stream.get("requests", [])) != 12 for stream in (*development, *held_out)):
        errors.append("request_count")
    if {stream.get("cohort") for stream in held_out} != set(COHORTS):
        errors.append("cohorts")
    if any(sum(stream.get("cohort") == cohort for stream in held_out) != 4 for cohort in COHORTS):
        errors.append("cohort_size")
    if len(live) != 24:
        errors.append("live_pair_count")
    for cohort in COHORTS:
        rows = [row for row in live if row.get("cohort") == cohort]
        if (
            len(rows) != 6
            or sum(row.get("presentation_order") == "original_first" for row in rows) != 3
        ):
            errors.append("live_balance")
    frozen = deepcopy(dict(public_manifest))
    observed = frozen.pop("manifest_hash", None)
    if observed != sha256_json(frozen):
        errors.append("public_manifest_hash")
    requests = [request for stream in (*development, *held_out) for request in stream["requests"]]
    if any(not 4 <= len(request["activities"]) <= 6 for request in requests):
        errors.append("activity_count")
    if len({sha256_json(request) for request in requests}) != len(requests):
        errors.append("unique_requests")
    return sorted(set(errors))


def _execute_process_boundary(
    repo_root: Path, paths: ExperimentPaths, *, progress: bool
) -> JsonDict:
    """Generate seals, run controls, then connect the public and private processes."""

    root = repo_root.resolve()
    python = str(root / ".venv/bin/python")
    private_script = str(root / "scripts/experiments/experiment_7330_v644_private_executor.py")
    paths.raw_root.mkdir(parents=True, exist_ok=True)
    manifest_receipt = _run_streamed(
        (
            python,
            "-u",
            private_script,
            "--build-manifests",
            "--public-manifest",
            str(paths.public_manifest),
            "--private-manifest",
            str(paths.private_manifest),
        ),
        root,
        paths.raw_root / "logs/build_manifests.log",
    )
    control_receipt = _run_streamed(
        (python, "-u", private_script, "--self-test"),
        root,
        paths.raw_root / "logs/executor_controls.log",
    )
    controls = json.loads(control_receipt["output_lines"][-1])
    _atomic_json(paths.executor_controls, controls)

    endpoint = Path("/tmp") / f"carnot-exp7330-{sha256_json(str(paths.raw_root))[-12:]}.sock"
    evaluator_argv = (
        python,
        "-u",
        private_script,
        "--serve",
        "--public-manifest",
        str(paths.public_manifest),
        "--private-manifest",
        str(paths.private_manifest),
        "--endpoint",
        str(endpoint),
        "--receipt",
        str(paths.evaluator_receipt),
    )
    evaluator_started = time.monotonic()
    print("[exp7330] phase=evaluator event=before_start", flush=True)
    evaluator = subprocess.Popen(  # noqa: S603 - fixed argument vector, never a shell.
        evaluator_argv,
        cwd=root,
        env=_environment(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    evaluator_lines: list[str] = []
    evaluator_events: queue.Queue[str] = queue.Queue()
    evaluator_thread = threading.Thread(
        target=_tee_reader,
        args=(evaluator, "exp7330-evaluator", evaluator_lines, evaluator_events),
        daemon=True,
    )
    evaluator_thread.start()
    deadline = time.monotonic() + 15.0
    while not endpoint.exists() and evaluator.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if not endpoint.exists():  # pragma: no cover - process startup failure guard.
        evaluator.terminate()
        raise RuntimeError("evaluator_endpoint_unavailable")
    print(
        f"[exp7330] phase=evaluator event=after_start pid={evaluator.pid} "
        f"elapsed_s={time.monotonic() - evaluator_started:.3f}",
        flush=True,
    )
    learner_receipt = _run_streamed(
        (
            python,
            "-u",
            "-m",
            "carnot.experiment_7330_v644_public_learner",
            "--public-manifest",
            str(paths.public_manifest),
            "--endpoint",
            str(endpoint),
            "--output",
            str(paths.learner_evidence),
        ),
        root,
        paths.raw_root / "logs/public_learner.log",
    )
    evaluator_exit = evaluator.wait(timeout=30)
    evaluator_thread.join(timeout=2)
    evaluator_log = paths.raw_root / "logs/private_evaluator.log"
    evaluator_log.parent.mkdir(parents=True, exist_ok=True)
    evaluator_log.write_text("".join(evaluator_lines), encoding="utf-8")
    print(
        f"[exp7330] phase=evaluator event=after_stop exit={evaluator_exit} "
        f"elapsed_s={time.monotonic() - evaluator_started:.3f}",
        flush=True,
    )
    if evaluator_exit != 0:  # pragma: no cover - child failure is retained by the caller.
        raise RuntimeError("evaluator_failed")
    return {
        "manifest_process": manifest_receipt,
        "control_process": control_receipt,
        "learner_process": learner_receipt,
        "evaluator_log_sha256": sha256_file(evaluator_log),
        "evaluator_exit_code": evaluator_exit,
        "evaluator_pid": evaluator.pid,
        "progress_enabled": progress,
    }


def independent_reduce(paths: ExperimentPaths) -> JsonDict:
    """Reload raw learner and evaluator rows without trusting artifact aggregates."""

    learner = _load_object(paths.learner_evidence)
    evaluator = _load_object(paths.evaluator_receipt)
    rows = learner["rows"]
    query_request_ids = [str(row["request_id"]) for row in evaluator["query_rows"]]
    held_out_or_live = sum(
        request_id.startswith("held-out-") or request_id.startswith("live-")
        for request_id in query_request_ids
    )
    return {
        "row_count": len(rows),
        "attempted_units": len(rows),
        "completed_units": sum(row.get("censored") is False for row in rows),
        "censored_units": sum(row.get("censored") is True for row in rows),
        "total_executor_calls": sum(int(row["executor_calls"]) for row in rows),
        "maximum_executor_calls": max(int(row["executor_calls"]) for row in rows),
        "maximum_state_bytes": max(int(row["state_bytes"]) for row in rows),
        "held_out_or_live_query_count": held_out_or_live,
        "query_count": len(evaluator["query_rows"]),
        "rows_sha256": sha256_json(rows),
    }


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    return {
        "expected": expected,
        "observed": observed,
        "pass": bool(passed),
        "passed": bool(passed),
        "principle": principle,
    }


def build_artifact(
    repo_root: Path,
    output_root: Path,
    *,
    progress: bool = True,
    precondition_overrides: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build real process evidence without running repository validation commands."""

    started = time.monotonic()
    checks, source_hashes = collect_preconditions(repo_root, overrides=precondition_overrides)
    if not gate_check_summary(checks)["passed"]:
        return build_blocked_artifact(checks, source_hashes)
    artifact = _base_artifact(checks, source_hashes)
    paths = ExperimentPaths.for_output_root(output_root)
    phase_started = time.monotonic()
    process_receipts = _execute_process_boundary(repo_root, paths, progress=progress)
    artifact["phase_spans"].append(
        {
            "phase": "manifest_controls_and_process_boundary",
            "duration_s": time.monotonic() - phase_started,
            "completed_units": 3,
            "checkpoint": str(paths.learner_evidence),
            "pending_operations": [],
        }
    )
    public_manifest = _load_object(paths.public_manifest)
    private_manifest = _load_object(paths.private_manifest)
    learner = _load_object(paths.learner_evidence)
    evaluator = _load_object(paths.evaluator_receipt)
    controls = _load_object(paths.executor_controls)
    reduction = independent_reduce(paths)
    manifest_errors = _manifest_errors(public_manifest)
    private_import_forbidden = any(
        "public_learner" in value or "experiment_7323" in value
        for value in evaluator["import_closure"]
    )
    process_separation = (
        len({os.getpid(), int(learner["learner_pid"]), int(evaluator["evaluator_pid"])}) == 3
    )
    learner_private_access_count = len(learner["forbidden_accesses"])
    response_keys = evaluator["response_keys"]
    boundary = {
        "claim_scope": "audited_process_boundary_not_hostile_process_security",
        "hostile_process_security_claim": False,
        "orchestrator_pid": os.getpid(),
        "learner_pid": learner["learner_pid"],
        "evaluator_pid": evaluator["evaluator_pid"],
        "process_separation_passed": process_separation,
        "learner_allowed_inputs": learner["allowed_input_paths"],
        "learner_allowed_outputs": learner["allowed_output_paths"],
        "learner_import_closure": learner["import_closure"],
        "learner_open_file_receipts": learner["open_file_receipts"],
        "learner_private_accesses": learner["forbidden_accesses"],
        "learner_private_access_count": learner_private_access_count,
        "evaluator_import_closure": evaluator["import_closure"],
        "evaluator_open_file_receipts": evaluator["open_file_receipts"],
        "evaluator_response_keys": response_keys,
        "evaluator_returned_private_fields": evaluator["returned_private_fields"],
        "held_out_or_live_query_count": reduction["held_out_or_live_query_count"],
        "private_executor_imports_learner_or_old_rules": private_import_forbidden,
        "process_receipts": process_receipts,
    }
    public_summary = {
        "path": str(paths.public_manifest),
        "public_sha256": sha256_file(paths.public_manifest),
        "public_manifest_hash": public_manifest["manifest_hash"],
        "evaluator_only_path": str(paths.private_manifest),
        "evaluator_only_sha256": sha256_file(paths.private_manifest),
        "evaluator_only_manifest_hash": private_manifest["manifest_hash"],
        "development_streams": len(public_manifest["development_streams"]),
        "held_out_streams": len(public_manifest["held_out_streams"]),
        "live_proposal_pairs": len(public_manifest["live_proposal_panel"]),
        "opaque_version_tokens_separate_from_rules": True,
        "rule_seed_disclosed_to_learner": False,
        "token_seed_disclosed_to_learner": False,
        "held_out_evaluator_labels_sealed": True,
        "private_label_count": private_manifest["label_count"],
    }
    witness_count = sum(
        bool(row["acceptance_witness"]["assignments"])
        for row in private_manifest["evaluator_records"].values()
    )
    all_witnesses = (
        private_manifest["all_acceptance_witnesses_nonempty"]
        and witness_count == private_manifest["label_count"]
    )
    compound = learner["compound_conflict_challenge"]
    gates = {
        "process_separation": _gate(
            True,
            process_separation,
            process_separation,
            "Learner and authority need distinct processes.",
        ),
        "independent_predicates": _gate(
            True,
            {
                "controls_passed": controls["passed"],
                "private_import_forbidden": private_import_forbidden,
            },
            controls["passed"] is True and not private_import_forbidden,
            "Executor semantics must not import the learner predicate or V643 rule helper.",
        ),
        "sealed_manifests": _gate(
            [],
            manifest_errors,
            not manifest_errors,
            "Fresh cohorts and live twins need exact seals.",
        ),
        "known_invalid_rejection": _gate(
            ">=5 malformed rejections",
            controls["malformed_rejection_count"],
            controls["malformed_rejection_count"] >= 5,
            "Malformed plans must fail closed.",
        ),
        "acceptance_region_witnesses": _gate(
            private_manifest["label_count"],
            witness_count,
            all_witnesses,
            "Every sealed private request needs one accepted witness.",
        ),
        "learner_private_access": _gate(
            0,
            learner_private_access_count,
            learner_private_access_count == 0,
            "The learner import and file receipts must not reach private rules.",
        ),
        "held_out_labels_unread": _gate(
            0,
            reduction["held_out_or_live_query_count"],
            reduction["held_out_or_live_query_count"] == 0,
            "Fixture construction cannot consume held-out or live labels.",
        ),
        "compound_conflict_conservative": _gate(
            {"full_rejected": True, "pair_projections_accepted": True, "learned_atoms": 0},
            compound,
            compound["full_rejected"]
            and compound["all_pair_projections_accepted"]
            and compound["learned_atom_count"] == 0,
            "Unsupported compound failures must not become pair atoms.",
        ),
        "development_rows_complete": _gate(
            {"rows": 48, "censored": 0},
            {"rows": reduction["row_count"], "censored": reduction["censored_units"]},
            reduction["row_count"] == 48 and reduction["censored_units"] == 0,
            "All fixed development units need terminal accounting.",
        ),
    }
    ready = all(row["passed"] for row in gates.values())
    raw_hashes = {
        "public_manifest": sha256_file(paths.public_manifest),
        "evaluator_private_manifest": sha256_file(paths.private_manifest),
        "learner_evidence": sha256_file(paths.learner_evidence),
        "evaluator_boundary_receipt": sha256_file(paths.evaluator_receipt),
        "executor_controls": sha256_file(paths.executor_controls),
    }
    artifact.update(
        {
            "rows": deepcopy(learner["rows"]),
            "sample_size_budget": {
                "planned_development_units": 48,
                "attempted_units": reduction["attempted_units"],
                "completed_units": reduction["completed_units"],
                "censored_units": reduction["censored_units"],
                "sealed_held_out_units_not_executed": 192,
                "reserved_live_proposal_pairs_not_executed": 24,
                "fixed_stopping_rule": "four development streams times twelve requests once",
            },
            "acceptance_gate_results": gates,
            "public_manifest": public_summary,
            "executor_boundary_receipt": boundary,
            "executor_controls": controls,
            "compound_conflict_challenge": compound,
            "independent_reduction": reduction,
            "executor_fixture_ready_score": int(ready),
            "source_artifact_hashes": {
                **artifact["source_artifact_hashes"],
                "raw_evidence": raw_hashes,
                "historical_diagnostic": {
                    "path": "results/experiment_7325_v643_addition_audit.json",
                    "sha256": artifact["source_artifact_hashes"]["current_sources"][
                        "results/experiment_7325_v643_addition_audit.json"
                    ],
                    "readiness_authority": False,
                },
            },
            "duration_s": time.monotonic() - started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    if not ready:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_executor_fixture_gate_failed"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _receipt_error(receipt: Mapping[str, Any]) -> bool:
    return not (
        isinstance(receipt.get("command"), str)
        and receipt.get("command")
        and isinstance(receipt.get("scope"), str)
        and isinstance(receipt.get("exit_code"), int)
        and isinstance(receipt.get("duration_s"), (int, float))
        and str(receipt.get("log_sha256", "")).startswith("sha256:")
    )


def validate_artifact(
    artifact: Mapping[str, Any], *, check_files: bool, require_validation: bool
) -> list[str]:
    """Cold-check schema, evidence, scores, receipts, and optional sidecars."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition:
            errors.append(name)

    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != 7330, "experiment_id")
    add(artifact.get("milestone") != "2026.09.644", "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("MODEL_SPECS") != [], "MODEL_SPECS")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    add(artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator", "substrate")
    add(
        artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator",
        "substrate_class",
    )
    add(artifact.get("execution_venue") != "host", "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    add(
        artifact.get("verdict_class")
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict_class",
    )
    add(
        not isinstance(artifact.get("field_principles"), Mapping)
        or any(key not in artifact["field_principles"] for key in _field_principles()),
        "field_principles",
    )
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    failed_class = artifact.get("verdict_class") in {"blocked", "disqualified"}
    add(
        failed_class
        and any(
            artifact.get(field) != 0
            for field in ("executor_fixture_ready_score", "executor_value_score", "promotion_score")
        ),
        "failed_scores",
    )
    if artifact.get("status") == "blocked":
        add(artifact.get("rows") != [], "blocked_rows")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "blocked_verdict")
        add(artifact.get("gate_check_summary", {}).get("first_failure") is None, "blocked_failure")
        return sorted(set(errors))
    add(artifact.get("status") != "complete", "status")
    rows = artifact.get("rows", [])
    add(not isinstance(rows, list) or len(rows) != 48, "row_count")
    add(any(int(row.get("executor_calls", 25)) > 24 for row in rows), "query_budget")
    add(any(int(row.get("state_bytes", 69_633)) > 69_632 for row in rows), "state_cap")
    gates = artifact.get("acceptance_gate_results", {})
    add(
        not isinstance(gates, Mapping)
        or any(
            not {"expected", "observed", "pass", "passed", "principle"} <= set(row)
            or row.get("pass") != row.get("passed")
            for row in gates.values()
        ),
        "acceptance_gate_results",
    )
    if artifact.get("executor_fixture_ready_score") == 1:
        add(artifact.get("verdict_class") != "circular_positive", "ready_verdict_class")
        add(
            not gates or not all(row.get("passed") is True for row in gates.values()), "ready_gates"
        )
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list) or any(_receipt_error(row) for row in receipts),
        "validation_receipts",
    )
    if require_validation:
        add(artifact.get("required_checks_passed") is not True, "required_checks_passed")
        passing_names = {row.get("name") for row in receipts if row.get("passed") is True}
        add(not set(REQUIRED_CHECK_NAMES) <= passing_names, "required_receipts")
    if check_files and artifact.get("status") == "complete":
        raw = artifact.get("source_artifact_hashes", {}).get("raw_evidence", {})
        for name, expected in raw.items():
            path = {
                "public_manifest": artifact["public_manifest"]["path"],
                "evaluator_private_manifest": artifact["public_manifest"]["evaluator_only_path"],
                "learner_evidence": artifact["executor_boundary_receipt"][
                    "learner_allowed_outputs"
                ][0],
                "evaluator_boundary_receipt": next(
                    value
                    for value in artifact["executor_boundary_receipt"][
                        "evaluator_open_file_receipts"
                    ]
                    if value.endswith("evaluator_boundary_receipt.json")
                )
                if any(
                    value.endswith("evaluator_boundary_receipt.json")
                    for value in artifact["executor_boundary_receipt"][
                        "evaluator_open_file_receipts"
                    ]
                )
                else None,
                "executor_controls": None,
            }[name]
            if path is None:
                continue
            add(sha256_file(Path(path)) != expected, f"raw_hash_{name}")
    return sorted(set(errors))


def write_artifact(
    path: Path, artifact: Mapping[str, Any], *, require_validation: bool = True
) -> JsonDict:
    """Validate a complete object before its one atomic terminal write."""

    errors = validate_artifact(
        artifact,
        check_files=artifact.get("status") == "complete",
        require_validation=require_validation,
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path)}


def run_terminal_validators(  # pragma: no cover - the declared entrypoint owns subprocess coverage.
    repo_root: Path, candidate: Path, log_dir: Path
) -> list[JsonDict]:
    """Run the two declared terminal readers against the measured candidate."""

    commands = [
        CommandSpec(
            "adversarial_verify",
            (
                str(repo_root / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                str(candidate),
            ),
            "measured_terminal_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(repo_root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_terminal_candidate",
        ),
    ]
    return run_commands(repo_root, commands, log_dir=log_dir)


def run_full_python_suite(  # pragma: no cover - the declared entrypoint owns subprocess coverage.
    repo_root: Path, log_dir: Path
) -> JsonDict:
    """Run the user-mandated Python suite once as current repository health."""

    command = CommandSpec(
        "full_python_suite",
        (str(repo_root / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_wide_required_observation",
        timeout_s=4_000.0,
    )
    return run_commands(repo_root, [command], log_dir=log_dir)[0]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - declared E2E entrypoint.
    args = _parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(
            _load_object(args.validate), check_files=True, require_validation=True
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    root = REPO_ROOT
    output_root = args.output_root.resolve() if args.output_root else root
    paths = ExperimentPaths.for_output_root(output_root)
    started = time.monotonic()
    print("[exp7330] phase=preconditions event=start", flush=True)
    artifact = build_artifact(root, output_root, progress=True)
    print(f"[exp7330] phase=fixture event=end status={artifact['status']}", flush=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.result, artifact, require_validation=False)
        print(f"[exp7330] phase=terminal_write event=end path={paths.result}", flush=True)
        return 0

    print("[exp7330] phase=scoped_validation event=start", flush=True)
    scoped_basetemp = Path("/tmp/carnot-exp7330-scoped")
    scoped_basetemp.mkdir(parents=True, exist_ok=True)
    scoped = run_scoped_validation(
        root,
        ["tests/python/test_experiment_7330_v644_executor_isolation.py"],
        [
            "python/carnot/experiment_7330_v644_public_learner.py",
            "python/carnot/experiment_7330_v644_executor_isolation.py",
        ],
        static_paths=[
            "scripts/experiments/experiment_7330_v644_private_executor.py",
            "scripts/experiments/experiment_7330_v644_executor_isolation.py",
        ],
        basetemp=scoped_basetemp,
        coverage_file=paths.raw_root / ".coverage",
        log_dir=paths.raw_root / "validation/scoped",
        historical_failures=_repository_health()["historical_failures"],
    )
    artifact.update(scoped)
    artifact["duration_s"] = time.monotonic() - started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _atomic_json(paths.candidate, artifact)
    print(
        f"[exp7330] phase=scoped_validation event=end passed={scoped['required_checks_passed']}",
        flush=True,
    )

    print("[exp7330] phase=terminal_validators event=start", flush=True)
    terminal_receipts = run_terminal_validators(
        root, paths.candidate, paths.raw_root / "validation/terminal"
    )
    artifact["validation_receipts"].extend(terminal_receipts)
    print("[exp7330] phase=terminal_validators event=end", flush=True)

    print("[exp7330] phase=full_python_suite event=start", flush=True)
    full_suite = run_full_python_suite(root, paths.raw_root / "validation/full_suite")
    artifact["repository_health"] = {
        **_repository_health(),
        "current_observation": full_suite,
        "status": "healthy" if full_suite["passed"] else "degraded_open",
        "affects_required_checks": not full_suite["passed"],
    }
    print(f"[exp7330] phase=full_python_suite event=end exit={full_suite['exit_code']}", flush=True)

    reduction = independent_reduce(paths)
    affected_failures = [row for row in artifact["validation_receipts"] if not row.get("passed")]
    if (
        not artifact["required_checks_passed"]
        or affected_failures
        or not full_suite["passed"]
        or reduction != artifact["independent_reduction"]
    ):
        artifact["executor_fixture_ready_score"] = 0
        artifact["executor_value_score"] = 0
        artifact["promotion_score"] = 0
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_current_validation_failed"
    artifact["duration_s"] = time.monotonic() - started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact,
        check_files=True,
        require_validation=artifact["verdict_class"] != "disqualified",
    )
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    print("[exp7330] phase=terminal_write event=start", flush=True)
    write_artifact(
        paths.result,
        artifact,
        require_validation=artifact["verdict_class"] != "disqualified",
    )
    print(f"[exp7330] phase=terminal_write event=end path={paths.result}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
