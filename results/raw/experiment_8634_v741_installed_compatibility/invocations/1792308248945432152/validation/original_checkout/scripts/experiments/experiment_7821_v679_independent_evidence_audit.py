#!/usr/bin/env python3
"""Run current V679 custody, reduction, and validation (REQ-REPORT-7821)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from carnot.experiment_7821_v679_independent_evidence_audit import (
    MANIFEST,
    build_artifact,
    cold_replay,
    inspect_sources,
    read_branches,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

FROZEN_MANIFEST_SHA256 = "sha256:0d37a89c8882177908a668ab0c4f85a0571546928c01f2b098212f89e7b7347f"
EXPECTED_NAMES = (
    "affected_pytest",
    "changed_code_coverage",
    "coverage_report",
    "ruff_check",
    "ruff_format",
    "mypy",
    "spec_coverage",
    "task_e2e",
    "fresh_reduction",
    "repository_health",
    "adversarial_verify",
    "strict_row_lint",
    "cold_replay",
)
Executor = Callable[[dict[str, Any], Path], tuple[int, bytes]]


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Expose real phase timing so a long child is visible to its operator."""
    print(
        f"phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def load_manifest(root: Path) -> dict[str, Any]:
    """Reject a changed or appended command before any child can start."""
    path = root / MANIFEST
    if sha256_file(path) != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen_manifest_hash_changed")
    value = json.loads(path.read_bytes())
    commands = value["commands"]
    if tuple(item["name"] for item in commands) != EXPECTED_NAMES:
        raise ValueError("undeclared_child")
    if any(
        item["classification"]
        != ("diagnostic" if item["name"] == "repository_health" else "required")
        for item in commands
    ):
        raise ValueError("command_class_changed")
    if len({item["private_root"] for item in commands}) != len(commands):
        raise ValueError("command_root_reused")
    return value


def execute_child(item: dict[str, Any], private: Path) -> tuple[int, bytes]:
    """Run one owned child with a deadline and heartbeat, then close its log."""
    log = private / "live.log"
    env = dict(os.environ, PYTHONPATH="python:.", JAX_PLATFORMS="cpu")
    exit_code = 124
    with log.open("wb") as stream:
        child = subprocess.Popen(item["argv"], stdout=stream, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + item["timeout_s"]
        while child.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                child.kill()
                child.wait()
                break
            try:
                child.wait(timeout=min(30, remaining))
            except subprocess.TimeoutExpired:
                print(
                    f"child={item['name']} event=heartbeat elapsed_s={time.monotonic():.3f}",
                    flush=True,
                )
        else:
            exit_code = child.returncode
    return exit_code, log.read_bytes()


def dispatch(
    root: Path, manifest: dict[str, Any], executor: Executor, durable_root: Path, start: float
) -> list[dict[str, Any]]:
    """Seal each closed child log to a private content-addressed path once."""
    receipts = []
    for index, item in enumerate(manifest["commands"]):
        if item["name"] != EXPECTED_NAMES[index]:
            raise ValueError("undeclared_child")
        private = Path(item["private_root"])
        private.mkdir(parents=True, exist_ok=True)
        for arg in item["argv"]:
            if arg.startswith("--basetemp="):
                Path(arg.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        progress(start, item["name"], "before_subprocess", index)
        exit_code, output = executor(item, private)
        progress(start, item["name"], "after_subprocess", index + 1)
        import hashlib

        digest = hashlib.sha256(output).hexdigest()
        sealed = durable_root / manifest["attempt_id"] / item["name"] / (digest + ".log")
        sealed.parent.mkdir(parents=True, exist_ok=True)
        if sealed.exists():
            if sealed.read_bytes() != output:
                raise ValueError("sealed_log_changed")
        else:
            temporary = sealed.with_suffix(".pending")
            if temporary.exists():
                raise ValueError("pending_log_reused")
            temporary.write_bytes(output)
            temporary.replace(sealed)
        receipts.append(
            {
                "name": item["name"],
                "argv": item["argv"],
                "classification": item["classification"],
                "exit_code": exit_code,
                "passed": exit_code == 0,
                "log_path": str(sealed),
                "log_sha256": sha256_file(sealed),
                "private_root": str(private),
            }
        )
    return receipts


def run_experiment(
    root: Path,
    date: str,
    output: Path,
    executor: Executor = execute_child,
    durable_root: Path | None = None,
) -> dict[str, Any]:
    """Build one candidate, run exact checks, and atomically publish the result."""
    root = root.resolve()
    start = time.monotonic()
    spans = []
    previous = 0.0

    def finish(phase: str, units: int) -> None:
        nonlocal previous
        now = time.monotonic() - start
        spans.append(
            {
                "phase": phase,
                "start_s": previous,
                "end_s": now,
                "duration_s": now - previous,
                "completed_units": units,
            }
        )
        previous = now
        progress(start, phase, "complete", units)

    progress(start, "preconditions", "start", 0)
    manifest = load_manifest(root)
    resources = {
        name: (root / name).is_file()
        for name in (".venv/bin/python", ".venv/bin/pytest", ".venv/bin/ruff", ".venv/bin/mypy")
    }
    sources, failures = inspect_sources(root)
    finish("preconditions", len(sources))
    progress(start, "reduction", "start", 0)
    branches, raw_failures = read_branches(root, sources)
    artifact = build_artifact(root, date, sources, failures + raw_failures, branches)
    artifact["preconditions_checked"]["resources"] = resources
    artifact["validation_receipts"]["frozen_affected_scope"] = {
        "tests": manifest["affected_tests"],
        "modules": manifest["changed_modules"],
        "rationale": manifest["scope_rationale"],
    }
    candidate = Path(manifest["private_root"]) / "candidate.json"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(candidate, artifact)
    finish("reduction", len(artifact["rows"]))
    progress(start, "validation", "start", 0)
    storage = (
        durable_root
        or root / "results/raw/experiment_7821_v679_independent_evidence_audit/validation"
    )
    receipts = dispatch(root, manifest, executor, storage, start)
    artifact["observed_child_commands"] = receipts
    artifact["repository_health"] = next(r for r in receipts if r["name"] == "repository_health")
    artifact["validation_receipts"]["commands"] = receipts
    artifact["validation_receipts"]["required_checks_passed"] = all(
        r["passed"] for r in receipts if r["classification"] == "required"
    )
    artifact["validation_receipts"]["candidate_sha256"] = sha256_file(candidate)
    artifact["validation_receipts"]["e2e_checks"] = [
        {
            "name": "task_e2e",
            "passed": next(r["passed"] for r in receipts if r["name"] == "task_e2e"),
        }
    ]
    adversarial = next(r for r in receipts if r["name"] == "adversarial_verify")
    try:
        report = json.loads(Path(adversarial["log_path"]).read_text())
        artifact["flagged_adversarial"] = bool(report.get("flagged_count", 1))
    except (OSError, ValueError, TypeError):
        artifact["flagged_adversarial"] = True
    if (
        not artifact["validation_receipts"]["required_checks_passed"]
        or artifact["flagged_adversarial"]
    ):
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["acceptance_gate_results"].update(validity=0, readiness=0)
        artifact["independent_evidence_ready_score"] = 0
    finish("validation", len(receipts))
    progress(start, "publication", "start", 0)
    finish("publication", 1)
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - start
    atomic_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Expose the real task entrypoint and a fresh-process cold reader."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args(argv)
    print("Exp7821 start elapsed_s=0.000 completed_units=0", flush=True)
    if args.cold is not None:
        failures = cold_replay(args.cold)
        print(json.dumps({"cold_replay_failures": failures}), flush=True)
        return int(bool(failures))
    root = Path(__file__).resolve().parents[2]
    result = run_experiment(
        root, args.date, root / "results/experiment_7821_v679_independent_evidence_audit.json"
    )
    print(
        json.dumps(
            {"honest_verdict": result["honest_verdict"], "verdict_class": result["verdict_class"]}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
