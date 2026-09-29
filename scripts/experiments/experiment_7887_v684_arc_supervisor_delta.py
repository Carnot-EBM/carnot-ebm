#!/usr/bin/env python3
"""Direct V684 ARC supervisor receipt reduction. REQ-REPORT-7887."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot.reporting.arc_supervisor_v683_delta import check_inputs, summarize
from carnot.reporting.arc_supervisor_v684_delta import aggregate, build_manifest, cold_replay
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "results/experiment_7860_v682_arc_supervisor_delta.json"
SOURCE = ROOT / "scripts/experiments/experiment_7860_v682_arc_supervisor_delta.py"
REGISTRY = ROOT / "ops/arc_solve_registry.yaml"
AGENT = ROOT / "python/carnot/agentic/arc_competition_agent.py"
OUTPUT = ROOT / "results/experiment_7887_v684_arc_supervisor_delta.json"
EXPECTED = (
    "sha256:02339c224e2e9c2bfce03d2da7bbc136e810a2b0f42a609c9a2c60a1e549bf68",
    "sha256:7e64814e631350118462b108fc947ebac845a25eff339387a85a507b2ed3424d",
    "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
)


def progress(started: float, phase: str, units: int) -> None:
    """Print each phase boundary with honest elapsed time."""

    print(
        f"[exp7887] phase={phase} elapsed_s={time.monotonic() - started:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def precheck() -> dict[str, Any]:
    """Pin known external bytes and inspect live policy reachability without a game."""

    checked = check_inputs(PRIOR, SOURCE, REGISTRY, EXPECTED)
    checks = checked["checks"]
    failures = checked["failures"]
    source = AGENT.read_text(encoding="utf-8") if AGENT.is_file() else ""
    tree = ast.parse(source) if source else ast.parse("")
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in (
        "_make_trajectory_supervisor",
        "_maybe_supervise_trajectory",
        "trajectory_supervisor_diagnostics",
    ):
        row = {
            "upstream_id": "arc_competition_agent",
            "path": str(AGENT),
            "sha256": sha256_file(AGENT) if AGENT.is_file() else None,
            "artifact_field": f"live_reachability.{name}",
            "op": "==",
            "expected": True,
            "observed": name in names,
            "role": "live_policy_reachability",
            "exposure_status": "read_only",
        }
        checks.append(row)
        if not row["observed"]:
            failures.append(row)
    checked["checks"] = checks
    checked["failures"] = failures
    return checked


def _seal(receipt: dict[str, Any], private: Path) -> dict[str, Any]:
    """Copy a closed child log to its immutable content addressed path."""

    original = Path(receipt["log_path"])
    digest = sha256_file(original).removeprefix("sha256:")
    sealed = private / "sealed" / f"{receipt['name']}_{digest}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.exists() and sealed.read_bytes() != original.read_bytes():
        raise ValueError("sealed_log_collision")
    if not sealed.exists():
        shutil.copyfile(original, sealed)
    receipt["log_path"] = str(sealed)
    receipt["log_sha256"] = "sha256:" + digest
    return receipt


def _run(private: Path, row: dict[str, Any], started: float) -> dict[str, Any]:
    """Run one frozen argv with a heartbeat and retain the actual exit."""

    progress(started, "before_" + row["name"], 0)
    spec = CommandSpec(row["name"], tuple(row["argv"]), row["classification"], row["deadline_s"])
    receipt = run_commands(ROOT, [spec], log_dir=private / "logs" / row["name"], heartbeat_s=30)[0]
    receipt = _seal(receipt, private)
    receipt["class"] = row["classification"]
    progress(started, "after_" + row["name"], int(receipt["passed"]))
    return receipt


def _artifact(
    date: str,
    started: float,
    checked: dict[str, Any],
    delta: dict[str, Any],
    private: Path,
    manifest: Path,
    cutoff: Path,
) -> dict[str, Any]:
    """Keep exact source custody, historical debt, and unmeasured benefit distinct."""

    prior = json.loads(PRIOR.read_text(encoding="utf-8")) if PRIOR.is_file() else {}
    source_paths = (
        PRIOR,
        SOURCE,
        REGISTRY,
        AGENT,
        ROOT / "python/carnot/reporting/arc_supervisor_v683_delta.py",
        ROOT / "python/carnot/reporting/arc_supervisor_v684_delta.py",
        ROOT / "tests/python/test_arc_supervisor_delta_7874.py",
        ROOT / "tests/python/test_arc_supervisor_delta_7887.py",
        Path(__file__).resolve(),
        manifest,
        cutoff,
    )
    hashes = {str(path): sha256_file(path) for path in source_paths if path.is_file()}
    hashes.update(
        {
            str(row["source_path"]): row["source_sha256"]
            for row in delta["outcome_rows"]
            if row.get("source_path") and row.get("source_sha256")
        }
    )
    blocked = bool(checked["failures"])
    elapsed = time.monotonic() - started
    artifact: dict[str, Any] = {
        "experiment_id": 7887,
        "task_id": "exp7887-arc-supervisor-delta",
        "milestone": "2026.09.684",
        "run_date": date,
        "honest_verdict": "complete_blocked_source_precondition"
        if blocked
        else "complete_null_no_new_supervisor_outcomes",
        "verdict_class": "blocked" if blocked else "null",
        "flagged_adversarial": False,
        "gate_check_summary": checked["failures"],
        "rows": delta["outcome_rows"],
        "sample_size_budget": delta["sample_size_budget"],
        "acceptance_gate_results": {
            "validity": None if blocked else True,
            "readiness": 0 if blocked else 1,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": elapsed,
        "phase_spans": [{"phase": "precheck_and_reduce", "duration_s": elapsed}],
        "random_seed": 0,
        "reproducibility_checksum": canonical_hash(
            {"sources": hashes, "seed": 0, "cutoff": checked["cutoff_ns"]}
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checked["checks"],
        "resolved_imports": {},
        "validation_receipts": [],
        "validation_command_manifest_path": str(manifest),
        "observed_child_commands": [],
        "historical_required_failures": prior.get("historical_required_failures", []),
        "repository_health": {
            "status": "unmeasured",
            "historical_full_suite_obligations_open": True,
        },
        "verifier_is_oracle": False,
        "claim_scope": "exposed_development; observational live receipt delta",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "file_hashes": []},
        "trained_head_specs": [],
        "historical_model_identity": "historical only; no current model calls",
        "arc_delta_ready_score": 0 if blocked else 1,
        "new_live_outcome_count": delta["new_live_outcome_count"],
        "no_new_outcomes": delta["no_new_outcomes"],
        "outcome_rows": delta["outcome_rows"],
        "cutoff_manifest_path": str(cutoff),
        "registry_sha256": hashes.get(str(REGISTRY)),
        "per_game_results": delta["per_game_results"],
        "new_level_solves": delta["new_level_solves"],
        "solve_provenance": [row.get("solve_provenance") for row in delta["outcome_rows"]],
        "firings": delta["firings"],
        "recommendation_rows": delta["recommendation_rows"],
        "validation_errors": [],
    }
    artifact["field_principles"] = {
        key: "Keep exact current evidence and leave unmeasured benefit unknown." for key in artifact
    }
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run direct private modes or publish the exact terminal checked bytes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--reduce-ledger", type=Path)
    parser.add_argument("--cutoff-ns", type=int)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "start", 0)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    if args.reduce_ledger is not None:
        if args.cutoff_ns is None or args.output is None:
            parser.error("--reduce-ledger needs --cutoff-ns and --output")
        progress(started, "before_private_reduce", 0)
        delta = aggregate(summarize(args.reduce_ledger, args.cutoff_ns, set(), {}))
        atomic_json(args.output, delta)
        progress(started, "after_private_reduce", delta["new_live_outcome_count"])
        return 0

    private = Path(tempfile.mkdtemp(prefix="carnot-exp7887-", dir="/tmp"))
    progress(started, "before_precheck", 0)
    checked = precheck()
    progress(started, "after_precheck", len(checked["checks"]))
    manifest_rows = build_manifest(ROOT, private, checked["cutoff_ns"])
    manifest = private / "validation_command_manifest.json"
    cutoff = private / "cutoff_manifest.json"
    atomic_json(
        manifest,
        {
            "commands": manifest_rows,
            "affected_closure": [
                "python/carnot/reporting/arc_supervisor_v684_delta.py",
                "scripts/experiments/experiment_7887_v684_arc_supervisor_delta.py",
                "tests/python/test_arc_supervisor_delta_7874.py",
                "tests/python/test_arc_supervisor_delta_7887.py",
            ],
            "scope_rationale": "V684 reader, its direct CLI and existing V683 consumer regressions",
            "inapplicable_e2e": ["E2E-009", "E2E-011", "E2E-013", "E2E-014"],
            "applicable_e2e": ["E2E-017"],
        },
    )
    atomic_json(
        cutoff,
        {
            "upstream_id": "exp7860",
            "path": str(PRIOR),
            "sha256": sha256_file(PRIOR) if PRIOR.is_file() else "missing",
            "cutoff_ns": checked["cutoff_ns"],
            "registry_sha256": sha256_file(REGISTRY) if REGISTRY.is_file() else "missing",
        },
    )
    progress(started, "frozen_manifest", len(manifest_rows))
    progress(started, "before_reduce", 0)
    delta = aggregate(
        summarize(
            ROOT if not checked["failures"] else private,
            checked["cutoff_ns"],
            checked["prior_hashes"],
            checked["registry_levels"],
        )
    )
    progress(started, "after_reduce", delta["new_live_outcome_count"])
    artifact = _artifact(args.date, started, checked, delta, private, manifest, cutoff)
    receipts: list[dict[str, Any]] = []
    if not checked["failures"]:
        for row in manifest_rows:
            receipt = _run(private, row, started)
            receipts.append(receipt)
            if row["name"] == "cli_failure_coverage":
                receipt["passed"] = receipt["exit_code"] != 0
            if row["name"] == "affected_pytest":
                artifact["resolved_imports"] = {
                    "carnot.reporting.arc_supervisor_v684_delta": str(
                        ROOT / "python/carnot/reporting/arc_supervisor_v684_delta.py"
                    )
                }
        artifact["validation_receipts"] = receipts
        artifact["observed_child_commands"] = [row["command_argv"] for row in receipts]
        artifact["validation_errors"] = [
            row["name"] for row in receipts if row["class"] == "required" and not row["passed"]
        ]
        if artifact["validation_errors"]:
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["arc_delta_ready_score"] = 0
            artifact["acceptance_gate_results"].update(validity=False, readiness=0)
    candidate = private / "terminal_candidate.json"
    artifact["duration_s"] = time.monotonic() - started
    artifact["phase_spans"].append(
        {
            "phase": "validation",
            "duration_s": artifact["duration_s"] - artifact["phase_spans"][0]["duration_s"],
        }
    )
    atomic_json(candidate, artifact)
    if not checked["failures"]:
        terminal = [
            {
                "name": "terminal_adversarial",
                "argv": [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(candidate),
                ],
                "deadline_s": 60,
                "classification": "required",
            },
            {
                "name": "terminal_rows",
                "argv": [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
                "deadline_s": 60,
                "classification": "required",
            },
        ]
        for attempt in range(2):
            reports = [_run(private, row, started) for row in terminal]
            atomic_json(private / f"terminal_reports_{attempt}.json", {"reports": reports})
            if all(row["passed"] for row in reports):
                break
            artifact["flagged_adversarial"] = not reports[0]["passed"]
            artifact["honest_verdict"] = "complete_disqualified_terminal_verification"
            artifact["verdict_class"] = "disqualified"
            artifact["arc_delta_ready_score"] = 0
            artifact["acceptance_gate_results"].update(validity=False, readiness=0)
            artifact["validation_errors"] = sorted(
                set(
                    artifact["validation_errors"]
                    + [row["name"] for row in reports if not row["passed"]]
                )
            )
            atomic_json(candidate, artifact)
    target = args.output or OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(candidate.read_bytes())
    progress(started, "deliverable_written", delta["new_live_outcome_count"])
    return 0 if artifact["arc_delta_ready_score"] == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
