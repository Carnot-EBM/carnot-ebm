"""Current bounded fixture and durable-bank qualification for REQ-VERIFY-7784.

The fixture checks execution mechanics. It cannot establish natural accuracy.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_7760_v675_online_runner as online
from carnot import experiment_7769_v676_training_qualification as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7784_v677_training_runtime.json"
PRIVATE = Path("/tmp/exp7784-private")
SCOPE = PRIVATE / "frozen_scope.json"
TESTS = (
    "tests/python/test_experiment_7784_v677_training_runtime.py",
    "tests/python/test_experiment_7769_v676_training_qualification.py",
    "tests/python/test_experiment_7755_v675_training_runtime.py",
    "tests/python/test_experiment_7760_v675_online_runner.py",
    "tests/python/test_experiment_7756_v675_evidence_view_protocol.py",
    "tests/python/test_experiment_7730_v673_set_energy_fit.py",
)
PRINCIPLES = {
    "experiment_id": "Each record needs a unique owner.",
    "milestone": "Bind the current research cycle.",
    "run_date": "Bind the measured run date.",
    "honest_verdict": "Unchanged external inputs must not consume retries.",
    "verdict_class": "Claim strength travels with the result.",
    "flagged_adversarial": "Invalid evidence cannot open a downstream gate.",
    "gate_check_summary": "A missing producer differs from a scientific null.",
    "rows": "A headline must be reproducible from individual units.",
    "acceptance_gate_results": "Working fixtures do not establish benefit.",
    "duration_s": "Real elapsed work determines substrate authenticity.",
    "phase_spans": "Validation and cold replay consume real time.",
    "random_seed": "A replay needs the same seed.",
    "reproducibility_checksum": "Another process must recover the same inputs.",
    "sample_size_budget": "Repeated views do not create families.",
    "source_artifact_hashes": "Old files cannot replace current producers.",
    "preconditions_checked": "Cheap failures precede compute.",
    "validation_receipts": "Every required check must pass before readiness.",
    "verifier_is_oracle": "Fixture truth is circular evidence.",
    "claim_scope": "Fixture success does not imply natural benefit.",
    "inference_substrate": "The invoked work sets duration expectations.",
    "inference_substrate_class": "No model is loaded in this task.",
    "MODEL_SPECS": "Only current model loads belong here.",
    "model_specs": "Only current model loads belong here.",
    "model_invocation_counts": "Count real calls, files, and tokens.",
    "training_runtime_ready_score": "Numeric checks and coverage must pass.",
    "online_runtime_ready_score": "Durable bank and query boundaries must pass.",
    "training_protocol_path": "A later trainer needs the frozen recipe.",
    "online_protocol_path": "A later runner needs the frozen recipe.",
    "coverage_shard_rows": "Every assertion must remain in bounded coverage.",
}


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Flush measured boundaries so a stalled child has a visible owner."""
    print(
        f"[exp7784] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def exercise_query_modes(folder: Path, names: list[str]) -> list[dict[str, Any]]:
    """Observe admitted-template versions while the real bank saves each query."""
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for mode in ("queued_commit", "between_query_commit"):
        runner = online.OnlineRunner(
            folder / mode / "state.json", folder / mode / "bank.json", names, "adaptive"
        )
        for block in range(8):
            if mode == "between_query_commit" and block:
                runner.release_block(block - 1, block)
            before = canonical_hash(runner.bank.state["templates"])
            runner.predict_block(block)
            after = canonical_hash(runner.bank.state["templates"])
            pending = bool(runner.state["queue"])
            if mode == "queued_commit":
                runner.release_block(block, block + 1)
            rows.append(
                {
                    "mode": mode,
                    "query": block,
                    "template_hash_before": before,
                    "template_hash_after": after,
                    "pending_feedback": pending,
                    "predictions_before_feedback": all(
                        row["label"] is None
                        for row in runner.state["rows"]
                        if row["block"] == block
                    )
                    if mode == "between_query_commit"
                    else pending,
                    "bank_path": str(runner.bank_path),
                    "state_path": str(runner.state_path),
                    "bank_hash": runner.bank.state_hash,
                }
            )
            progress(started, "online_query", mode, len(rows))
        if mode == "between_query_commit":
            runner.release_block(7, 8)
        runner.finish()
    return rows


def cold_reduce(candidate: Path) -> dict[str, Any]:
    """Read saved rows anew so a changed source byte fails the terminal gate."""
    value = json.loads(candidate.read_text())
    rows = json.loads(Path(value["raw_paths"]["rows"]).read_text())
    if rows != value["rows"]:
        raise ValueError("raw_rows_invalid")
    if "fixture_training_rows" in value:
        prior.cold_reduce(candidate)
    return {
        "valid": True,
        "row_count": len(rows),
        "raw_sha256": sha256_file(Path(value["raw_paths"]["rows"])),
    }


def current_record(
    measured: dict[str, Any],
    date: str,
    failures: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    shards: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    """Carry exact fixture rows forward while setting only current gate scores."""
    valid = (
        not failures
        and measured["online_fixture"]["valid"]
        and all(row["template_hash_before"] == row["template_hash_after"] for row in query_rows)
    )
    record = {**measured}
    record.update(
        {
            "experiment_id": 7784,
            "milestone": "2026.09.677",
            "run_date": date,
            "honest_verdict": "complete_circular_positive_training_runtime"
            if valid
            else "complete_disqualified_required_validation",
            "verdict_class": "circular_positive" if valid else "disqualified",
            "flagged_adversarial": False,
            "gate_check_summary": failures,
            "acceptance_gate_results": {
                "validity": valid,
                "readiness": int(valid),
                "probability_quality": None,
                "decision_benefit": None,
                "retention": None,
                "efficiency": None,
            },
            "duration_s": duration,
            "field_principles": PRINCIPLES,
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "inference_substrate_class": "no_model_load",
            "planned_inference_substrate_class": "no_model_load",
            "MODEL_SPECS": [],
            "model_specs": [],
            "model_invocation_counts": {"calls": 0, "tokens": 0, "loaded_files": []},
            "training_runtime_ready_score": int(valid),
            "online_runtime_ready_score": int(valid),
            "coverage_shard_rows": shards,
            "query_commit_rows": query_rows,
            "historical_exp7769_verdict": "complete_disqualified_required_validation",
        }
    )
    record["reproducibility_checksum"] = canonical_hash(
        {
            "date": date,
            "seed": measured.get("random_seed"),
            "code": sha256_file(Path(__file__)),
            "rows": sha256_file(Path(measured["raw_paths"]["rows"]))
            if measured.get("raw_paths")
            else None,
            "scope": sha256_file(SCOPE) if SCOPE.is_file() else None,
            "protocol": sha256_file(Path(measured["training_protocol_path"])),
            "online_protocol": sha256_file(Path(measured["online_protocol_path"])),
        }
    )
    record["validation_receipts"] = {
        **measured.get("validation_receipts", {}),
        "frozen_affected_scope": json.loads(SCOPE.read_text()) if SCOPE.is_file() else {},
        "coverage_shards": shards,
    }
    return record


def supervise(
    name: str, argv: list[str], log: Path, timeout_s: float, start: float
) -> dict[str, Any]:
    """Poll one owned child and save its exact exit and log bytes."""
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": "python:.",
            "PYTHONUNBUFFERED": "1",
            "JAX_PLATFORMS": "cpu",
            "COVERAGE_CORE": "sysmon",
        }
    )
    progress(start, name, "before_subprocess", 0)
    began = time.monotonic()
    with log.open("wb") as stream:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                exit_code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                progress(start, name, "subprocess_outstanding", 0)
                if time.monotonic() - began >= timeout_s:
                    child.terminate()
                    try:
                        exit_code = child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        exit_code = child.wait()
                    break
    elapsed = time.monotonic() - began
    progress(start, name, "after_subprocess", 1)
    return {
        "name": name,
        "command_argv": argv,
        "exit_code": exit_code,
        "duration_s": elapsed,
        "timed_out": elapsed >= timeout_s,
        "passed": exit_code == 0 and elapsed < timeout_s,
        "log_path": str(log),
        "log_sha256": sha256_file(log),
    }


def failure(receipt: dict[str, Any]) -> dict[str, Any]:
    """Keep the literal failed operand beside its exact log hash."""
    return {
        "upstream_id": "exp7784",
        "artifact_path": receipt["log_path"],
        "artifact_hash": receipt["log_sha256"],
        "field": f"{receipt['name']}.exit_code",
        "operator": "==",
        "expected": 0,
        "observed": receipt["exit_code"],
    }


def blocked_record(date: str, checks: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """Keep all planned fixture units when a declared external input fails."""
    failed = [
        {
            "upstream_id": row["upstream_id"],
            "artifact_path": row["artifact_path"],
            "artifact_hash": row.get("artifact_sha256"),
            "field": row["field"],
            "operator": row["operator"],
            "expected": row["expected"],
            "observed": row["observed"],
        }
        for row in checks
        if not row["passed"]
    ]
    rows = [
        {
            "family": f"fixture-{i}",
            "arm": arm,
            "status": "unstarted_external_precondition",
            "excluded": True,
            "censored": False,
            "raw_path": None,
        }
        for arm in prior.qualification.ARMS
        for i in range(6)
    ]
    return {
        "experiment_id": 7784,
        "milestone": "2026.09.677",
        "run_date": date,
        "honest_verdict": "complete_blocked_external_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": rows,
        "acceptance_gate_results": {
            "validity": False,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": duration,
        "phase_spans": [
            {"phase": "preflight", "duration_s": duration, "completed_units": len(checks)}
        ],
        "random_seed": prior.runtime.SEEDS[0],
        "reproducibility_checksum": canonical_hash(checks),
        "sample_size_budget": {
            "intended": 6,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "excluded": 6,
            "censored": 0,
            "effective_independent_n": 0,
        },
        "source_artifact_hashes": [
            {
                "upstream_id": row["upstream_id"],
                "artifact_path": row["artifact_path"],
                "artifact_hash": row.get("artifact_sha256"),
                "run_date": None,
                "imported_fields": {row["field"]: row["observed"]},
                "eligibility": False,
            }
            for row in checks
        ],
        "preconditions_checked": {"online_bank_checks": checks},
        "validation_receipts": {"frozen_affected_scope": json.loads(SCOPE.read_text())},
        "verifier_is_oracle": True,
        "claim_scope": "unstarted_fixture_external_block",
        "field_principles": PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"calls": 0, "tokens": 0, "loaded_files": []},
        "training_runtime_ready_score": 0,
        "online_runtime_ready_score": 0,
        "training_protocol_path": None,
        "online_protocol_path": None,
        "coverage_shard_rows": [],
    }


def run(date: str) -> dict[str, Any]:
    """Measure current mechanics, then gate the exact fixture candidate."""
    started = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    progress(started, "preflight", "before", 0)
    checks, _, names = online.preflight(ROOT)
    old = ROOT / "results/experiment_7769_v676_training_qualification.json"
    checks.append(
        {
            "upstream_id": "exp7769",
            "artifact_path": str(old.relative_to(ROOT)),
            "artifact_sha256": sha256_file(old) if old.is_file() else None,
            "field": "exists",
            "operator": "==",
            "expected": True,
            "observed": old.is_file(),
            "passed": old.is_file(),
        }
    )
    failed = [row for row in checks if not row["passed"]]
    progress(started, "preflight", "after", len(checks))
    if failed:
        artifact = blocked_record(date, checks, time.monotonic() - started)
        candidate = PRIVATE / "blocked_candidate.json"
        atomic_json(candidate, artifact)
        readers = [
            supervise(
                "adversarial_verify",
                [sys.executable, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)],
                PRIVATE / "logs/adversarial_verify.log",
                60,
                started,
            ),
            supervise(
                "strict_row_consistency",
                [
                    sys.executable,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
                PRIVATE / "logs/strict_row_consistency.log",
                60,
                started,
            ),
        ]
        artifact["validation_receipts"]["terminal_readers"] = readers
        report = (
            json.loads(Path(readers[0]["log_path"]).read_text()) if readers[0]["passed"] else {}
        )
        artifact["flagged_adversarial"] = bool(report.get("flagged_count", 1))
        artifact["gate_check_summary"].extend(failure(row) for row in readers if not row["passed"])
        artifact["duration_s"] = time.monotonic() - started
        atomic_json(OUTPUT, artifact)
        progress(started, "publish", "blocked_atomic_write", len(artifact["rows"]))
        return artifact
    phase = time.monotonic()
    progress(started, "fixture", "before_benchmark", 0)
    measured = prior.run_fixture(PRIVATE / "fixture", date)
    measured["preconditions_checked"] = {
        **measured["preconditions_checked"],
        "current_online_bank_checks": checks,
    }
    progress(started, "fixture", "after_benchmark", len(measured["fixture_training_rows"]))
    spans = [{"phase": "fixture", "duration_s": time.monotonic() - phase, "completed_units": 9}]
    phase = time.monotonic()
    query_rows = exercise_query_modes(PRIVATE / "query_modes", names)
    spans.append(
        {
            "phase": "online_query",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(query_rows),
        }
    )
    protocol = Path(measured["online_protocol_path"])
    online_value = json.loads(protocol.read_text())
    online_value["query_commit_modes"] = ["queued_commit", "between_query_commit"]
    online_value["within_query_template_version_immutable"] = True
    atomic_json(protocol, online_value)
    artifact = current_record(measured, date, [], query_rows, [], time.monotonic() - started)
    artifact["phase_spans"] = [*measured["phase_spans"], *spans]
    artifact["preconditions_checked"] = {
        **measured["preconditions_checked"],
        "current_online_bank_checks": checks,
    }
    artifact["source_artifact_hashes"].append(
        {
            "upstream_id": "exp7769",
            "artifact_path": str(old.relative_to(ROOT)),
            "artifact_hash": sha256_file(old),
            "run_date": json.loads(old.read_text())["run_date"],
            "imported_fields": {"honest_verdict": json.loads(old.read_text())["honest_verdict"]},
            "eligibility": False,
        }
    )
    candidate = PRIVATE / "candidate.json"
    atomic_json(candidate, artifact)
    receipts: list[dict[str, Any]] = []
    phase = time.monotonic()
    receipts.append(
        supervise(
            "cold_replay",
            [
                sys.executable,
                "-u",
                "scripts/experiments/experiment_7784_v677_training_runtime.py",
                "--cold-reduce",
                str(candidate),
            ],
            PRIVATE / "logs/cold_replay.log",
            120,
            started,
        )
    )
    spans.append(
        {
            "phase": "cold_replay",
            "duration_s": time.monotonic() - phase,
            "completed_units": int(receipts[-1]["passed"]),
        }
    )
    shards = (TESTS[:2], TESTS[2:5], TESTS[5:])
    coverage_dir = PRIVATE / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    basetemp = PRIVATE / "basetemp"
    basetemp.mkdir(parents=True, exist_ok=True)
    include = "*/experiment_7784_v677_training_runtime.py"
    phase = time.monotonic()
    for index, paths in enumerate(shards):
        data = coverage_dir / f".coverage.shard{index}"
        (basetemp / f"shard{index}").parent.mkdir(parents=True, exist_ok=True)
        row = supervise(
            f"coverage_shard_{index}",
            [
                str(ROOT / ".venv/bin/coverage"),
                "run",
                f"--data-file={data}",
                f"--include={include}",
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={basetemp / f'shard{index}'}",
                *paths,
                "-q",
            ],
            PRIVATE / f"logs/coverage_shard_{index}.log",
            299,
            started,
        )
        row["data_path"] = str(data)
        row["data_hash"] = sha256_file(data) if data.is_file() else None
        receipts.append(row)
    spans.append(
        {"phase": "coverage_shards", "duration_s": time.monotonic() - phase, "completed_units": 3}
    )
    combined = coverage_dir / ".coverage"
    receipts.append(
        supervise(
            "coverage_combine",
            [
                str(ROOT / ".venv/bin/coverage"),
                "combine",
                "--keep",
                f"--data-file={combined}",
                str(coverage_dir),
            ],
            PRIVATE / "logs/coverage_combine.log",
            60,
            started,
        )
    )
    receipts.append(
        supervise(
            "coverage_report",
            [
                str(ROOT / ".venv/bin/coverage"),
                "report",
                f"--data-file={combined}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ],
            PRIVATE / "logs/coverage_report.log",
            60,
            started,
        )
    )
    receipts.append(
        supervise(
            "affected_pytest",
            [
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={basetemp / 'affected'}",
                *TESTS,
                "-q",
            ],
            PRIVATE / "logs/affected_pytest.log",
            299,
            started,
        )
    )
    for name, argv in (
        (
            "ruff_check",
            [
                str(ROOT / ".venv/bin/ruff"),
                "check",
                "python/carnot/experiment_7784_v677_training_runtime.py",
                "scripts/experiments/experiment_7784_v677_training_runtime.py",
                *TESTS,
            ],
        ),
        (
            "ruff_format",
            [
                str(ROOT / ".venv/bin/ruff"),
                "format",
                "--check",
                "python/carnot/experiment_7784_v677_training_runtime.py",
                "scripts/experiments/experiment_7784_v677_training_runtime.py",
                *TESTS,
            ],
        ),
        (
            "mypy",
            [
                str(ROOT / ".venv/bin/mypy"),
                "python/carnot/experiment_7784_v677_training_runtime.py",
            ],
        ),
        ("spec_coverage", [sys.executable, "-u", "scripts/check_spec_coverage.py", *TESTS]),
        (
            "full_python_suite",
            [
                str(ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={basetemp / 'full'}",
            ],
        ),
    ):
        receipts.append(
            supervise(
                name,
                argv,
                PRIVATE / f"logs/{name}.log",
                600 if name == "full_python_suite" else 90,
                started,
            )
        )
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
        }
    )
    failures = [failure(row) for row in receipts if not row["passed"]]
    artifact = current_record(
        measured,
        date,
        failures,
        query_rows,
        [row for row in receipts if row["name"].startswith("coverage_")],
        time.monotonic() - started,
    )
    artifact["phase_spans"] = [*measured["phase_spans"], *spans]
    artifact["validation_receipts"]["commands"] = receipts
    artifact["validation_receipts"]["exact_coverage_replay"] = {
        "argv_log": "/tmp/exp7784-repro/coverage-verbose.log",
        "log_sha256": sha256_file(Path("/tmp/exp7784-repro/coverage-verbose.log")),
    }
    artifact["validation_receipts"]["broad_repository_collection"] = receipts[-1]
    artifact["coverage_combined_report"] = {
        "data_path": str(combined),
        "data_hash": sha256_file(combined) if combined.is_file() else None,
        "report": receipts[5],
    }
    atomic_json(candidate, artifact)
    phase = time.monotonic()
    terminal = [
        supervise(
            "adversarial_verify",
            [sys.executable, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)],
            PRIVATE / "logs/adversarial_verify.log",
            60,
            started,
        ),
        supervise(
            "strict_row_consistency",
            [
                sys.executable,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ],
            PRIVATE / "logs/strict_row_consistency.log",
            60,
            started,
        ),
    ]
    artifact["validation_receipts"]["terminal_readers"] = terminal
    report = json.loads(Path(terminal[0]["log_path"]).read_text()) if terminal[0]["passed"] else {}
    artifact["flagged_adversarial"] = bool(report.get("flagged_count", 1))
    artifact["gate_check_summary"].extend(failure(row) for row in terminal if not row["passed"])
    if artifact["flagged_adversarial"] or artifact["gate_check_summary"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["training_runtime_ready_score"] = 0
        artifact["online_runtime_ready_score"] = 0
        artifact["acceptance_gate_results"]["readiness"] = 0
        artifact["acceptance_gate_results"]["validity"] = False
    artifact["phase_spans"].append(
        {
            "phase": "terminal_readers",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(terminal),
        }
    )
    artifact["duration_s"] = time.monotonic() - started
    atomic_json(OUTPUT, artifact)
    progress(started, "publish", "after_atomic_write", len(artifact["rows"]))
    return artifact
