#!/usr/bin/env python3
"""Publish the V685 authority lifecycle receipt. REQ-REPORT-7891-V685."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from carnot.reporting.current_work_receipt import atomic_json, sha256_file  # noqa: E402
from carnot.reporting.v685_authority_lifecycle import assess_authorities, cold_replay  # noqa: E402

START = time.monotonic()
MODEL_SPECS: list[str] = []
RESULT = ROOT / "results/experiment_7891_v685_authority_lifecycle.json"
RAW = ROOT / "results/raw/experiment_7891_v685_authority_lifecycle"
DESIGN = ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
STAGED = ROOT / "research-roadmap-next.yaml"
ACTIVE = ROOT / "research-roadmap.yaml"
OWNED = [
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "scripts/experiments/experiment_7891_v685_authority_lifecycle.py",
]
AFFECTED = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_experiment_7837_v681_contract_methods.py",
    "tests/python/test_experiment_7877_v683_independent_audit.py",
    "tests/python/test_experiment_7878_v683_capstone.py",
    "tests/python/test_experiment_7879_v684_contract_methods.py",
    "tests/python/test_experiment_7890_v684_capstone.py",
]
CONSUMERS = [
    "tests/python/test_roadmap_schema.py",
    "tests/python/test_audit_roadmap_gates.py",
    "tests/python/test_exclusion_manifest_lint.py",
    "tests/python/test_conductor_gates.py",
]


def progress(phase: str, event: str, units: int) -> None:
    """Keep live work visible to the conductor and give real elapsed time."""
    print(
        f"[exp7891] phase={phase} event={event} elapsed_s={time.monotonic() - START:.3f} completed_units={units}",
        flush=True,
    )


def command_manifest(output: Path, negative_candidate: Path, negative_raw: Path) -> dict[str, Any]:
    """Freeze exact argv and expected exits before any owned child starts."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    paths = OWNED + [
        "python/carnot/reporting/v683_independent_audit.py",
        "python/carnot/reporting/v683_capstone.py",
        "scripts/experiments/experiment_7837_v681_contract_methods.py",
        "scripts/experiments/experiment_7879_v684_contract_methods.py",
    ]
    return {
        "version": 1,
        "affected_tests": AFFECTED,
        "consumer_regressions": CONSUMERS,
        "coverage_includes": [str(ROOT / path) for path in OWNED],
        "commands": [
            {
                "name": "negative_cli_replay",
                "argv": [
                    py,
                    __file__,
                    "--cold-replay",
                    str(negative_candidate),
                    "--raw",
                    str(negative_raw),
                ],
                "expected_exit": 1,
                "deadline_s": 60,
            },
            {
                "name": "affected_pytest",
                "argv": [pytest, "-n", "0", "-o", "addopts=", "--no-cov", "-q", *AFFECTED],
                "expected_exit": 0,
                "deadline_s": 900,
            },
            {
                "name": "consumer_pytest",
                "argv": [pytest, "-n", "0", "-o", "addopts=", "--no-cov", "-q", *CONSUMERS],
                "expected_exit": 0,
                "deadline_s": 240,
            },
            {
                "name": "ruff_check",
                "argv": [ruff, "check", *paths, *AFFECTED],
                "expected_exit": 0,
                "deadline_s": 90,
            },
            {
                "name": "ruff_format",
                "argv": [ruff, "format", "--check", *paths, *AFFECTED],
                "expected_exit": 0,
                "deadline_s": 90,
            },
            {
                "name": "mypy_strict",
                "argv": [mypy, "--strict", *paths],
                "expected_exit": 0,
                "deadline_s": 180,
            },
            {
                "name": "spec_coverage",
                "argv": [py, "scripts/check_spec_coverage.py", *AFFECTED],
                "expected_exit": 0,
                "deadline_s": 180,
            },
        ],
        "inapplicable_e2e": {
            "E2E-015": "No source producer changed",
            "E2E-016": "No intervention or Qwen path changed",
            "E2E-017": "No ARC receipt reducer changed",
        },
        "output": str(output),
    }


def run_child(spec: dict[str, Any], index: int, raw_root: Path) -> dict[str, Any]:
    """Supervise one owned child and seal its log only after the child exits."""
    name = spec["name"]
    argv = spec["argv"]
    progress(name, "before_subprocess", index)
    raw_root.mkdir(parents=True, exist_ok=True)
    temp = raw_root / f"{name}.running.log"
    started = time.monotonic()
    env = {**os.environ, "PYTHONPATH": "python:.", "JAX_PLATFORMS": "cpu"}
    with temp.open("wb") as stream:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
        timed_out = False
        while child.poll() is None:
            try:
                child.wait(
                    timeout=min(30.0, max(0.1, spec["deadline_s"] - (time.monotonic() - started)))
                )
            except subprocess.TimeoutExpired:
                progress(name, "heartbeat", index)
            if time.monotonic() - started >= spec["deadline_s"] and child.poll() is None:
                child.kill()
                child.wait()
                timed_out = True
    digest = sha256_file(temp)
    sealed = raw_root / "sealed_logs" / f"{name}-{digest.removeprefix('sha256:')}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if not sealed.exists():
        shutil.copyfile(temp, sealed)
    temp.unlink()
    progress(name, "after_subprocess", index + 1)
    return {
        "name": name,
        "argv": argv,
        "expected_exit": spec["expected_exit"],
        "actual_exit": child.returncode,
        "deadline_s": spec["deadline_s"],
        "duration_s": time.monotonic() - started,
        "timed_out": timed_out,
        "passed": not timed_out and child.returncode == spec["expected_exit"],
        "log_path": str(sealed),
        "log_sha256": sha256_file(sealed),
    }


def mutation_controls(args: argparse.Namespace, raw_root: Path) -> list[dict[str, Any]]:
    """Challenge the same contract using private files, including full-task fields."""
    baseline = yaml.safe_load(args.active.read_text())
    controls = [
        ("id", 0),
        ("title", 0),
        ("phase", 0),
        ("deliverable", 0),
        ("MODEL_SPECS", 0),
        ("inference_substrate_class", 0),
        ("gated_on", 3),
        ("prompt", 0),
        ("prior_failure", 0),
        ("old_active", 0),
        ("missing_staging", 0),
        ("later_staging", 0),
    ]
    rows = []
    for index, (name, task_index) in enumerate(controls):
        private = raw_root / "mutations" / name
        private.mkdir(parents=True, exist_ok=True)
        active = private / "active.yaml"
        staged = private / "staged.yaml"
        value = deepcopy(baseline)
        if name == "prior_failure":
            value["tasks"][0]["prior_failures"][0]["verdict"] = "changed"
        elif name == "old_active":
            value["milestone"] = "2026.09.684"
        elif name not in {"missing_staging", "later_staging"}:
            value["tasks"][task_index][name] = "changed"
        active.write_text(yaml.safe_dump(value, sort_keys=False))
        if name == "later_staging":
            staged.write_text("milestone: 2026.09.686\ntasks: []\n")
        result = assess_authorities(args.design, staged, active, private / "snapshots")
        expected = name in {"missing_staging", "later_staging"}
        rows.append(
            {
                "unit_id": name,
                "arm": "authority_mutation",
                "status": "completed",
                "expected_activation": expected,
                "observed_activation": result["activated"],
                "passed": result["activated"] is expected,
            }
        )
        progress("mutation_controls", name, index + 1)
    return rows


def build_candidate(args: argparse.Namespace, raw_root: Path) -> dict[str, Any]:
    """Reduce actual authority bytes and current checks without science claims."""
    progress("preconditions", "start", 0)
    assessed = assess_authorities(
        args.design, args.staged, args.active, raw_root / "authority_snapshots"
    )
    progress("preconditions", "complete", 3)
    negative_raw = raw_root / "negative_cli_replay_rows.json"
    negative_candidate = raw_root / "negative_cli_replay_candidate.json"
    rows = assessed["contract_rows"]
    tampered_rows = deepcopy(rows)
    tampered_rows[0]["absolute_metric"] = 2
    atomic_json(negative_raw, {"rows": rows})
    atomic_json(
        negative_candidate,
        {"experiment_id": 7891, "task_id": "exp7891-authority-lifecycle", "rows": tampered_rows},
    )
    manifest = command_manifest(args.output, negative_candidate, negative_raw)
    manifest_path = raw_root / "validation_command_manifest.json"
    atomic_json(manifest_path, manifest)
    receipts = (
        []
        if args.no_validation
        else [run_child(spec, index, raw_root) for index, spec in enumerate(manifest["commands"])]
    )
    owned_passed = bool(receipts) and all(item["passed"] for item in receipts)
    activated = assessed["activated"]
    ready = int(activated and owned_passed)
    if not activated:
        verdict, verdict_class = "complete_blocked_authority", "blocked"
    elif args.no_validation:
        verdict, verdict_class = "partial_unvalidated_private", "partial"
    elif not owned_passed:
        verdict, verdict_class = "complete_disqualified_required_validation", "disqualified"
    else:
        verdict, verdict_class = "complete_circular_positive_authority", "circular_positive"
    mutations = mutation_controls(args, raw_root)
    failed = sum(row["status"] == "completed" and not row["matched"] for row in rows)
    completed = sum(row["status"] == "completed" for row in rows)
    source_paths = [
        (args.design, "design_authority"),
        (args.staged, "staged_authority"),
        (args.active, "active_authority"),
        (ROOT / "research-references.md", "exposed_method_review"),
        (ROOT / "results/experiment_7879_v684_contract_methods.json", "historical_failure"),
        (ROOT / "results/experiment_7890_v684_capstone.json", "historical_failure"),
    ]
    sources = [
        {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "role": role,
            "exposure_status": "historical"
            if role == "historical_failure"
            else "exposed_development",
        }
        for path, role in source_paths
    ]
    prior = ROOT / "results/experiment_7890_v684_capstone.json"
    old = json.loads(prior.read_text()) if prior.is_file() else {}
    historical = old.get("historical_required_failures", [])
    literature = [
        {
            "source": "TOOD",
            "method": "paired temporal drift controls",
            "decision": "adapt",
            "measured_benefit": None,
        },
        {
            "source": "Continual Calibration",
            "method": "separate retention of uncertainty",
            "decision": "adapt",
            "measured_benefit": None,
        },
        {
            "source": "SURE-RAG and Verification Without Sufficiency",
            "method": "whole-source and witness controls",
            "decision": "adapt",
            "measured_benefit": None,
        },
        {
            "source": "Constrained decoding and semantic gap",
            "method": "separate syntax, fidelity and semantics",
            "decision": "adapt",
            "measured_benefit": None,
        },
        {
            "source": "Delayed online optimization",
            "method": "feedback release times",
            "decision": "adapt",
            "measured_benefit": None,
        },
        {
            "source": "VeriFin",
            "method": "source identity before arithmetic checks",
            "decision": "defer",
            "measured_benefit": None,
        },
    ]
    candidate: dict[str, Any] = {
        "experiment_id": 7891,
        "task_id": "exp7891-authority-lifecycle",
        "milestone": "2026.09.685",
        "run_date": args.date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": assessed["gate_check_summary"],
        "rows": rows,
        "contract_rows": rows,
        "sample_size_budget": {
            "intended": 12,
            "eligible": 12,
            "started": 12,
            "completed": completed,
            "failed": failed,
            "censored": 0,
            "excluded": failed,
            "independent": completed - failed,
        },
        "acceptance_gate_results": {
            "validity": ready,
            "readiness": ready,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - START,
        "phase_spans": [
            {
                "phase": "authority_and_validation",
                "duration_s": time.monotonic() - START,
                "completed_units": 3 + len(receipts),
            }
        ],
        "random_seed": 6857891,
        "reproducibility_checksum": assessed["canonical_tasks_sha256"][:16],
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "design_exists": args.design.is_file(),
            "active_exists": args.active.is_file(),
            "staged_exists": args.staged.is_file(),
            "active_milestone_matches": activated,
            "design_digest_matches_active": assessed["active_tasks_sha256"]
            == assessed["canonical_tasks_sha256"],
        },
        "resolved_imports": {
            "carnot.reporting.v685_authority_lifecycle": str(ROOT / OWNED[0]),
            "scripts.experiments.experiment_7891_v685_authority_lifecycle": str(
                Path(__file__).resolve()
            ),
        },
        "validation_receipts": receipts,
        "validation_command_manifest_path": str(manifest_path),
        "observed_child_commands": [item["argv"] for item in receipts],
        "historical_required_failures": historical,
        "repository_health": old.get("repository_health", {"status": "historical_unavailable"}),
        "verifier_is_oracle": True,
        "claim_scope": {
            "authority": "administrative_contract",
            "science": "unmeasured",
            "exposure": "exposed_development",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [],
        "target_model": "none",
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0},
        "trained_head_specs": [],
        "contract_ready_score": ready,
        "mutation_rows": mutations,
        "authority_snapshots": assessed["authority_snapshots"],
        "canonical_tasks_sha256": assessed["canonical_tasks_sha256"],
        "literature_adoption_decisions": literature,
        "planning_matched": assessed["planning_matched"],
        "activation_confirmed": activated,
    }
    candidate["field_principles"] = {
        key: "Record actual task-owned evidence; authority agreement cannot establish scientific benefit."
        for key in candidate
    }
    candidate["field_principles"]["field_principles"] = "Explain each task-owned field and gate."
    return candidate


def terminal_validate(candidate: Path, raw_root: Path) -> tuple[bool, list[dict[str, Any]]]:
    """Run both validators on the exact final candidate and retain their reports."""
    checks = [
        (
            "adversarial_verify",
            [sys.executable, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(candidate)],
        ),
        (
            "verdict_row_consistency",
            [
                sys.executable,
                str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ],
        ),
    ]
    reports = []
    for index, (name, argv) in enumerate(checks):
        receipt = run_child(
            {"name": name, "argv": argv, "expected_exit": 0, "deadline_s": 120}, index, raw_root
        )
        report_path = (
            raw_root
            / "validator_reports"
            / f"{name}-{sha256_file(candidate).removeprefix('sha256:')}.json"
        )
        atomic_json(
            report_path,
            {
                "candidate_path": str(candidate),
                "candidate_sha256": sha256_file(candidate),
                **receipt,
            },
        )
        reports.append(
            {**receipt, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}
        )
    return all(report["passed"] for report in reports), reports


def main() -> int:
    """Handle private fixture paths and publish only checked default bytes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--design", type=Path, default=DESIGN)
    parser.add_argument("--staged", type=Path, default=STAGED)
    parser.add_argument("--active", type=Path, default=ACTIVE)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--raw", type=Path, default=RAW / "rows.json")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()
    progress("startup", "start", 0)
    if args.cold_replay:
        ok = cold_replay(args.cold_replay, args.raw)
        if not ok:
            print("cold_replay_mismatch", flush=True)
        progress("cold_replay", "complete", int(ok))
        return 0 if ok else 1
    if args.date != "20260929":
        parser.error("run_date must be 20260929")
    raw_root = args.raw.parent
    candidate = build_candidate(args, raw_root)
    atomic_json(args.raw, {"rows": candidate["rows"]})
    temporary = raw_root / "terminal_candidate.json"
    atomic_json(temporary, candidate)
    if not args.no_validation:
        valid, reports = terminal_validate(temporary, raw_root)
        if not valid:
            candidate["flagged_adversarial"] = any(
                item["name"] == "adversarial_verify" and item["actual_exit"] != 0
                for item in reports
            )
            candidate["contract_ready_score"] = 0
            candidate["acceptance_gate_results"]["readiness"] = 0
            candidate["honest_verdict"] = "complete_disqualified_terminal_validation"
            candidate["verdict_class"] = "disqualified"
            atomic_json(temporary, candidate)
            valid, reports = terminal_validate(temporary, raw_root)
        atomic_json(
            raw_root / "terminal_validator_reports.json",
            {"candidate_sha256": sha256_file(temporary), "reports": reports, "passed": valid},
        )
        if not valid:
            raise ValueError("terminal_recheck_failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, candidate)
    if sha256_file(args.output) != sha256_file(temporary):
        raise ValueError("published bytes differ from checked candidate")
    progress("publication", "complete", 12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
