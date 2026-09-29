#!/usr/bin/env python3
"""Audit new ARC supervisor receipts without replaying solved levels.

REQ-REPORT-7860: only authenticated live agent receipts after the prior
snapshot can affect this observational ledger. The CLI never starts a game.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any

from carnot.reporting.arc_supervisor_receipt_delta import reduce_ledger
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "results/experiment_7845_v681_arc_supervisor_delta.json"
SOURCE = ROOT / "scripts/experiments/experiment_7845_v681_arc_supervisor_delta.py"
REGISTRY = ROOT / "ops/arc_solve_registry.yaml"
DELIVERABLE = ROOT / "results/experiment_7860_v682_arc_supervisor_delta.json"
EXPECTED = {
    PRIOR: "sha256:d5c947106843371192fb72d2ccb526bf065174163b35221b828d30db85a08deb",
    SOURCE: "sha256:0048e46881707bf003101e62bb7e23124ddb160b4d9cfd70d78e18330744b9a6",
    REGISTRY: "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
}


def progress(started: float, phase: str, units: int) -> None:
    """Make each bounded phase visible to the operator and process supervisor."""

    print(
        f"[exp7860] phase={phase} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def preconditions() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, set[str]]:
    """Freeze exact source bytes and report missing operands separately."""

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path, expected in EXPECTED.items():
        observed = sha256_file(path) if path.is_file() else "missing"
        check = {
            "upstream_id": path.stem,
            "path": str(path),
            "sha256": observed if observed != "missing" else None,
            "artifact_field": "sha256",
            "op": "==",
            "expected": expected,
            "observed": observed,
            "role": "registry_precheck" if path == REGISTRY else "historical_cutoff_only",
            "exposure_status": "read_only",
        }
        checks.append(check)
        if observed != expected:
            failures.append(check)
    if failures:
        return checks, failures, 0, set()
    text = REGISTRY.read_text(encoding="utf-8")
    version = re.search(r"^schema_version:\s*(\d+)\s*$", text, re.M)
    observed_version = int(version.group(1)) if version else None
    version_check = {
        "upstream_id": "arc_solve_registry",
        "path": str(REGISTRY),
        "sha256": sha256_file(REGISTRY),
        "artifact_field": "schema_version",
        "op": "==",
        "expected": 1,
        "observed": observed_version,
        "role": "registry_precheck",
        "exposure_status": "read_only",
    }
    checks.append(version_check)
    if observed_version != 1:
        failures.append(version_check)
    levels_present = bool(re.search(r"^\s*levels_reproduced:\s*[1-9]\d*\s*$", text, re.M))
    levels_check = {
        "upstream_id": "arc_solve_registry",
        "path": str(REGISTRY),
        "sha256": sha256_file(REGISTRY),
        "artifact_field": "levels_reproduced_positive",
        "op": "==",
        "expected": True,
        "observed": levels_present,
        "role": "registry_precheck",
        "exposure_status": "read_only",
    }
    checks.append(levels_check)
    if not levels_present:
        failures.append(levels_check)
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    prior_class = prior.get("verdict_class")
    class_check = {
        "upstream_id": "exp7845-arc-supervisor-delta",
        "path": str(PRIOR),
        "sha256": sha256_file(PRIOR),
        "artifact_field": "verdict_class",
        "op": "==",
        "expected": "disqualified",
        "observed": prior_class,
        "role": "historical_validation_only",
        "exposure_status": "disqualified",
    }
    checks.append(class_check)
    if prior_class != "disqualified":
        failures.append(class_check)
    hashes = {
        value
        for value in prior.get("source_artifact_hashes", {}).values()
        if isinstance(value, str)
    }
    cutoff = max(PRIOR.stat().st_mtime_ns, SOURCE.stat().st_mtime_ns)
    return checks, failures, cutoff, hashes


def _commands(private: Path) -> list[dict[str, Any]]:
    """Fix validation argv and deadlines before a receipt can enter the delta."""

    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    module = "python/carnot/reporting/arc_supervisor_receipt_delta.py"
    cli = "scripts/experiments/experiment_7860_v682_arc_supervisor_delta.py"
    tests = "tests/python/test_arc_supervisor_receipt_delta_7860.py"
    return [
        {
            "name": "worktree_imports",
            "argv": [
                py,
                "-c",
                "import carnot.reporting.arc_supervisor_receipt_delta as m; import json; print(json.dumps({'resolved_imports': {'carnot.reporting.arc_supervisor_receipt_delta': m.__file__}}))",
            ],
            "classification": "required",
            "deadline_s": 30,
        },
        {
            "name": "affected_pytest",
            "argv": [
                pytest,
                tests,
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'basetemp'}",
            ],
            "classification": "required",
            "deadline_s": 90,
        },
        {
            "name": "changed_coverage",
            "argv": [
                cov,
                "run",
                f"--data-file={private / 'coverage.unit'}",
                f"--include=*/{module.removeprefix('python/')}",
                "-m",
                "pytest",
                tests,
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'covtemp'}",
            ],
            "classification": "required",
            "deadline_s": 90,
        },
        {
            "name": "coverage_report",
            "argv": [
                cov,
                "report",
                f"--data-file={private / 'coverage.unit'}",
                f"--include=*/{module.removeprefix('python/')}",
                "--show-missing",
                "--fail-under=100",
            ],
            "classification": "required",
            "deadline_s": 30,
        },
        {
            "name": "ruff_check",
            "argv": [ruff, "check", module, cli, tests],
            "classification": "required",
            "deadline_s": 30,
        },
        {
            "name": "ruff_format",
            "argv": [ruff, "format", "--check", module, cli, tests],
            "classification": "required",
            "deadline_s": 30,
        },
        {
            "name": "mypy",
            "argv": [mypy, "--strict", module, cli],
            "classification": "required",
            "deadline_s": 60,
        },
        {
            "name": "scoped_spec",
            "argv": [py, "scripts/check_spec_coverage.py", tests],
            "classification": "required",
            "deadline_s": 30,
        },
    ]


def _artifact(
    date: str,
    started: float,
    checks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    cutoff_path: Path,
    manifest_path: Path,
    delta: dict[str, Any],
) -> dict[str, Any]:
    """Keep zero activity and historical disqualification explicit in one record."""

    paths = [
        PRIOR,
        SOURCE,
        REGISTRY,
        cutoff_path,
        manifest_path,
        ROOT / "python/carnot/reporting/arc_supervisor_receipt_delta.py",
        ROOT / "python/carnot/reporting/arc_supervisor_delta.py",
        ROOT / "python/carnot/reporting/current_work_receipt.py",
        ROOT / "python/carnot/reporting/experiment_7303_validation_scope.py",
        ROOT / "python/carnot/agentic/arc_supervisor_refinement.py",
        ROOT / "python/carnot/agentic/arc_trajectory_supervisor.py",
        ROOT / "tests/python/test_arc_supervisor_receipt_delta_7860.py",
        Path(__file__).resolve(),
    ]
    sources = {str(path): sha256_file(path) for path in paths if path.is_file()}
    sources.update(
        {
            row["source_path"]: row["source_sha256"]
            for row in delta["outcome_rows"]
            if row.get("source_path") and row.get("source_sha256")
        }
    )
    elapsed = time.monotonic() - started
    verdict = (
        "complete_blocked_source_precondition"
        if failures
        else (
            "complete_null_no_new_supervisor_outcomes"
            if delta["no_new_outcomes"]
            else "complete_null_observational_supervisor_delta"
        )
    )
    rows = delta["outcome_rows"] or [
        {
            "unit": "inventory_delta",
            "arm": None,
            "source_family": "live_supervisor_receipt",
            "seed": None,
            "status": "completed",
            "intended": 0,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "censored": 0,
            "excluded": 0,
            "independent_n": 0,
        }
    ]
    artifact: dict[str, Any] = {
        "experiment_id": 7860,
        "task_id": "exp7860-arc-supervisor-delta",
        "milestone": "2026.09.682",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": "blocked" if failures else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "sample_size_budget": delta["sample_size_budget"],
        "acceptance_gate_results": {
            "validity": None if failures else True,
            "readiness": 0 if failures else 1,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": elapsed,
        "phase_spans": [{"phase": "preconditions_and_reduce", "duration_s": elapsed}],
        "random_seed": 0,
        "reproducibility_checksum": canonical_hash({"sources": sources, "seed": 0}),
        "source_artifact_hashes": sources,
        "preconditions_checked": checks,
        "validation_receipts": [],
        "validation_command_manifest_path": str(manifest_path),
        "observed_child_commands": [],
        "repository_health": {
            "status": "unmeasured",
            "historical_full_suite_obligations_open": True,
        },
        "verifier_is_oracle": False,
        "claim_scope": "observational live supervisor ledger; no causal action saving or new solve",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "file_hashes": []},
        "current_work_receipt": build_current_work_receipt(
            run_id=f"exp7860-{time.monotonic_ns()}",
            owner_pid=os.getpid(),
            events=[],
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_details={"source_count": len(delta["outcome_rows"])},
            inference_substrate_class="aggregation",
            execution_venue="host",
            started_monotonic_ns=time.monotonic_ns() - int(elapsed * 1_000_000_000),
            ended_monotonic_ns=time.monotonic_ns(),
        ),
        "arc_delta_ready_score": 0 if failures else 1,
        "no_new_outcomes": delta["no_new_outcomes"],
        "outcome_rows": delta["outcome_rows"],
        "cutoff_manifest_path": str(cutoff_path),
        "solve_provenance": [r.get("solve_provenance") for r in delta["outcome_rows"]],
        "new_level_solves": 0,
        "firings": delta["firings"],
        "recommendation_rows": delta["recommendation_rows"],
        "historical_required_failures": [
            {
                "upstream_id": "exp7845",
                "name": r["name"],
                "exit_code": r["exit_code"],
                "log_path": r["log_path"],
                "log_sha256": r["log_sha256"],
            }
            for r in json.loads(PRIOR.read_text())["validation_receipts"]
            if r["class"] == "required" and not r["passed"]
        ]
        if not failures
        else [],
    }
    artifact["field_principles"] = {
        key: "Preserve exact evidence, unknown values, and the boundary of this observational claim."
        for key in artifact
    }
    for key in artifact["acceptance_gate_results"]:
        artifact["field_principles"][f"acceptance_gate_results.{key}"] = (
            "A measured valid ledger does not establish scientific benefit."
        )
    return artifact


def _seal(receipt: dict[str, Any], private: Path) -> dict[str, Any]:
    """Copy closed child output to a path named by its exact content hash."""

    original = Path(receipt["log_path"])
    digest = sha256_file(original).removeprefix("sha256:")
    sealed = private / "sealed" / f"{receipt['name']}_{digest}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.is_file() and sealed.read_bytes() != original.read_bytes():
        raise ValueError("sealed_log_collision")
    if not sealed.is_file():
        shutil.copyfile(original, sealed)
    receipt["log_path"] = str(sealed)
    receipt["log_sha256"] = "sha256:" + digest
    return receipt


def cold_replay(candidate: Path) -> list[str]:
    """Rehash closed logs and independently recount primitive live firings."""

    document = json.loads(candidate.read_text(encoding="utf-8"))
    errors = [
        receipt["name"]
        for receipt in document.get("validation_receipts", [])
        if not Path(receipt["log_path"]).is_file()
        or sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
    ]
    actual = sum(
        row.get("fired") is True
        for row in document.get("outcome_rows", [])
        if row.get("status") in {"completed", "censored"}
    )
    if actual != document.get("firings"):
        errors.append("primitive_firing_count")
    return errors


def validate(
    artifact: dict[str, Any], private: Path, commands: list[dict[str, Any]], started: float
) -> None:
    """Run frozen owned commands and retain every failure without reclassifying it."""

    receipts: list[dict[str, Any]] = []
    candidate = private / "candidate.json"
    for index, command in enumerate(commands):
        if command["name"] in {"cold_replay", "adversarial_verify", "strict_rows"}:
            artifact["validation_receipts"] = receipts
            artifact["duration_s"] = time.monotonic() - started
            atomic_json(candidate, artifact)
        progress(started, f"before_{command['name']}", index)
        spec = CommandSpec(
            command["name"],
            tuple(command["argv"]),
            command["classification"],
            float(command["deadline_s"]),
        )
        found = run_commands(ROOT, [spec], log_dir=private / "logs" / str(index), heartbeat_s=30)
        receipt = _seal(found[0], private)
        receipt["class"] = command["classification"]
        receipts.append(receipt)
        progress(started, f"after_{command['name']}", index + 1)
    artifact["validation_receipts"] = receipts
    artifact["observed_child_commands"] = [row["command_argv"] for row in receipts]
    failures = [row["name"] for row in receipts if row["class"] == "required" and not row["passed"]]
    imports = next(
        row.get("resolved_imports") for row in receipts if row["name"] == "worktree_imports"
    )
    if (
        not isinstance(imports, dict)
        or not imports
        or any(
            not Path(path).resolve().is_relative_to(ROOT / "python") for path in imports.values()
        )
    ):
        failures.append("resolved_imports")
    artifact["validation_errors"] = failures
    artifact["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and not row["passed"] for row in receipts
    )
    valid = not failures and not artifact["gate_check_summary"]
    artifact["acceptance_gate_results"]["validity"] = valid
    artifact["acceptance_gate_results"]["readiness"] = int(valid)
    artifact["arc_delta_ready_score"] = int(valid)
    if artifact["gate_check_summary"]:
        artifact["honest_verdict"] = "complete_blocked_source_precondition"
        artifact["verdict_class"] = "blocked"
    elif failures:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    health = next(row for row in receipts if row["name"] == "repository_health_180s")
    artifact["repository_health"] = {
        "status": "healthy" if health["passed"] else "failed_health",
        "diagnostic": health,
        "historical_full_suite_obligations_open": True,
    }
    artifact["duration_s"] = time.monotonic() - started
    artifact["phase_spans"].append(
        {"phase": "owned_validation", "duration_s": artifact["duration_s"]}
    )
    atomic_json(candidate, artifact)


def main() -> int:
    """Run the read-only ledger or its frozen terminal validation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--reduce-ledger", type=Path)
    parser.add_argument("--cutoff-ns", type=int)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    progress(started, "start", 0)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    if args.reduce_ledger is not None:
        if args.cutoff_ns is None or args.output is None:
            parser.error("--reduce-ledger needs --cutoff-ns and --output")
        result = reduce_ledger(args.reduce_ledger, args.cutoff_ns, set())
        atomic_json(args.output, result)
        progress(started, "private_delta_written", len(result["outcome_rows"]))
        return 0
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7860-", dir="/tmp"))
    progress(started, "before_preconditions", 0)
    checks, failures, cutoff, prior_hashes = preconditions()
    progress(started, "after_preconditions", len(checks))
    commands = _commands(private)
    py = str(ROOT / ".venv/bin/python")
    commands.extend(
        [
            {
                "name": "cli_e2e",
                "argv": [
                    py,
                    __file__,
                    "--reduce-ledger",
                    str(ROOT),
                    "--cutoff-ns",
                    str(cutoff),
                    "--output",
                    str(private / "e2e.json"),
                ],
                "classification": "required",
                "deadline_s": 60,
            },
            {
                "name": "cold_replay",
                "argv": [py, __file__, "--cold-replay", str(private / "candidate.json")],
                "classification": "required",
                "deadline_s": 30,
            },
            {
                "name": "adversarial_verify",
                "argv": [
                    py,
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(private / "candidate.json"),
                ],
                "classification": "required",
                "deadline_s": 60,
            },
            {
                "name": "strict_rows",
                "argv": [
                    py,
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(private / "candidate.json"),
                ],
                "classification": "required",
                "deadline_s": 30,
            },
            {
                "name": "repository_health_180s",
                "argv": [
                    str(ROOT / ".venv/bin/pytest"),
                    "tests/python",
                    "-q",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private / 'health-temp'}",
                ],
                "classification": "diagnostic",
                "deadline_s": 180,
            },
        ]
    )
    manifest = private / "validation_command_manifest.json"
    atomic_json(
        manifest,
        {
            "commands": commands,
            "source_hashes": {
                str(path): sha256_file(path)
                for path in (
                    ROOT / "python/carnot/reporting/arc_supervisor_receipt_delta.py",
                    ROOT / "python/carnot/reporting/arc_supervisor_delta.py",
                    ROOT / "tests/python/test_arc_supervisor_receipt_delta_7860.py",
                    Path(__file__).resolve(),
                )
            },
            "inapplicable_e2e": "Full live ARC loop checks require a runtime consumer change; this is a reporting reader.",
        },
    )
    cutoff_path = private / "cutoff_manifest.json"
    atomic_json(
        cutoff_path,
        {
            "cutoff_ns": cutoff,
            "prior_artifact_sha256": EXPECTED[PRIOR],
            "prior_source_sha256": EXPECTED[SOURCE],
            "registry_sha256": EXPECTED[REGISTRY],
        },
    )
    progress(started, "frozen_manifest", len(commands))
    delta = (
        reduce_ledger(ROOT, cutoff, prior_hashes)
        if not failures
        else reduce_ledger(private, cutoff, set())
    )
    progress(started, "after_reduce", len(delta["outcome_rows"]))
    artifact = _artifact(args.date, started, checks, failures, cutoff_path, manifest, delta)
    validate(artifact, private, commands, started)
    atomic_json(args.output or DELIVERABLE, artifact)
    progress(started, "deliverable_written", len(delta["outcome_rows"]))
    return 0 if artifact["arc_delta_ready_score"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
