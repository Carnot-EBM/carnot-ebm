#!/usr/bin/env python3
"""Execute V681 contract custody and bounded validation; REQ-REPORT-7837."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import yaml  # noqa: E402

from carnot.reporting.current_work_receipt import (  # noqa: E402
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.roadmap_contract import (  # noqa: E402
    cold_replay,
    compare_contract,
    snapshot_authorities,
    verify_snapshots,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7837_v681_contract_methods"
MANIFEST = RAW / "validation_command_manifest.json"
MANIFEST_SHA256 = "643a588a631779924e700e8f9866194cf49e924859d24b69999439d797f764ab"
RESULT = ROOT / "results/experiment_7837_v681_contract_methods.json"
DESIGN = ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
STAGED = ROOT / "research-roadmap-next.yaml"
ACTIVE = ROOT / "research-roadmap.yaml"
SNAPSHOTS = ROOT / "docs/research-notes/v681-authority-snapshots"
METHOD = ROOT / "docs/research-notes/v681-method-map.md"
REQUIRED_SOURCES = (
    DESIGN,
    STAGED,
    ACTIVE,
    METHOD,
    ROOT / "research-references.md",
    ROOT / "research-complete.yaml",
    ROOT / "ops/exclusion_manifest.yaml",
    ROOT / "results/experiment_7823_v680_contract_methods.json",
    ROOT / "results/experiment_7824_v680_source_feature_isolation.json",
    ROOT / "results/experiment_7828_v680_counter_evidence_protocol.json",
    ROOT / "results/experiment_7831_v680_arc_supervisor_refinement.json",
    ROOT / "results/experiment_7834_v680_hardware_evidence.json",
    ROOT / "results/experiment_7835_v680_independent_evidence_audit.json",
    ROOT / "results/experiment_7836_v680_capstone.json",
    ROOT / "python/carnot/reporting/roadmap_contract.py",
    ROOT / "tests/python/test_experiment_7837_v681_contract_methods.py",
    Path(__file__),
    MANIFEST,
)
START = time.monotonic()


def progress(phase: str, event: str, units: int) -> None:
    """Print a flushed elapsed-time boundary for the owning process."""
    print(
        f"[exp7837] phase={phase} event={event} elapsed_s={time.monotonic() - START:.3f} completed_units={units}",
        flush=True,
    )


def seal_log(path: Path, name: str) -> dict[str, str]:
    """Copy a closed child log into a unique content-addressed durable path."""
    digest = sha256_file(path)
    target = RAW / "sealed_logs" / f"{name}-{digest.removeprefix('sha256:')}.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) != digest:
        raise ValueError("sealed log hash collision")
    if not target.exists():
        shutil.copyfile(path, target)
    try:
        label = str(target.relative_to(ROOT))
    except ValueError:
        label = str(target)
    return {"path": label, "sha256": digest}


def run_child(command: dict[str, Any], index: int) -> dict[str, Any]:
    """Supervise only this owned child, with a deadline and 30-second heartbeat."""
    name = str(command["name"])
    argv = [str(part) for part in command["argv"]]
    timeout_s = float(command["timeout_s"])
    progress(name, "before_subprocess", index)
    started = time.monotonic()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH="python:.")
    with tempfile.NamedTemporaryFile(dir=RAW, suffix=".log", delete=False) as stream:
        scratch = Path(stream.name)
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= timeout_s:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            try:
                process.wait(timeout=min(30.0, max(0.1, timeout_s - elapsed)))
            except subprocess.TimeoutExpired:
                progress(name, "heartbeat", index)
        exit_code = process.wait()
    seal = seal_log(scratch, name)
    output = scratch.read_text(errors="replace")
    scratch.unlink()
    receipt: dict[str, Any] = {
        "name": name,
        "command_argv": argv,
        "classification": command["classification"],
        "timeout_s": timeout_s,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": time.monotonic() - started,
        "log_path": seal["path"],
        "log_sha256": seal["sha256"],
        "passed": exit_code == 0 and not timed_out,
        "output_tail": output[-2000:],
    }
    if name == "worktree_imports":
        try:
            parsed = json.loads(output.strip().splitlines()[-1])["resolved_imports"]
        except (IndexError, ValueError, KeyError):
            parsed = {}
        receipt["resolved_imports"] = parsed
        receipt["passed"] = (
            receipt["passed"]
            and bool(parsed)
            and all(Path(path).is_relative_to(ROOT / "python") for path in parsed.values())
        )
    progress(name, f"after_subprocess_exit_{exit_code}", index + 1)
    return receipt


def base_artifact(
    date: str,
    comparison: dict[str, Any],
    sources: list[dict[str, Any]],
    snapshots: list[dict[str, str]],
    missing: list[Path],
) -> dict[str, Any]:
    """Construct the evidence fields before any validating child can run."""
    rows = comparison["rows"]
    for row in rows:
        row["raw_paths"] = [item["path"] for item in snapshots]
    gate_failures = [
        {
            "upstream": "preconditions",
            "path": str(path),
            "sha256": None,
            "artifact_field": "exists",
            "op": "==",
            "expected": True,
            "observed": False,
        }
        for path in missing
    ]
    source_hashes = [item for item in sources if item["sha256"]]
    value: dict[str, Any] = {
        "experiment_id": 7837,
        "task_id": "exp7837-contract-methods",
        "milestone": "2026.09.681",
        "run_date": date,
        "honest_verdict": "complete_partial_v681_validation_pending",
        "verdict_class": "partial",
        "flagged_adversarial": False,
        "gate_check_summary": gate_failures,
        "rows": rows,
        "sample_size_budget": {
            "intended": 14,
            "eligible": len(rows),
            "started": len(rows),
            "completed": sum(row["status"] == "completed" for row in rows),
            "censored": 0,
            "excluded": sum(row["excluded"] for row in rows),
            "independent": 0,
            "independent_unit": "none_administrative",
        },
        "acceptance_gate_results": {
            "validity": False,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": 0.0,
        "phase_spans": {},
        "random_seed": None,
        "reproducibility_checksum": canonical_hash(
            {"sources": source_hashes, "seeds": [68101, 68102, 68103]}
        ),
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "source_paths": sources,
            "resource_ownership": "host_cpu_only_no_gpu_lease",
            "gate_operands": [],
            "development_roles": {
                "fit": 256,
                "tune": 64,
                "policy": 64,
                "online_update": 96,
                "online_admission": 64,
                "evaluation": 64,
                "retention": 32,
            },
            "missing": [str(path) for path in missing],
            "historical_exp7823_is_not_a_science_prerequisite": True,
        },
        "validation_receipts": [],
        "validation_command_manifest_path": str(MANIFEST.relative_to(ROOT)),
        "validation_command_manifest_sha256": sha256_file(MANIFEST),
        "observed_child_commands": [],
        "repository_health": None,
        "verifier_is_oracle": True,
        "claim_scope": {
            "kind": "administrative_exact_authority",
            "science": "unmeasured",
            "RAGTruth": "all_640_families_exposed_development",
            "fresh_generalization_eligible": False,
        },
        "field_principles": {
            "experiment_id/task_id": "Keep numeric experiment identity separate from the task slug.",
            "honest_verdict/verdict_class": "A terminal class travels with its evidence.",
            "flagged_adversarial": "Reader flags quarantine the result.",
            "gate_check_summary": "Missing and wrong-valued operands are distinct.",
            "rows/sample_size_budget": "Raw units let a reader recompute counts; views and seeds do not raise N.",
            "acceptance_gate_results.validity": "Source and label custody is necessary.",
            "acceptance_gate_results.readiness": "Runnable qualified interfaces are necessary.",
            "acceptance_gate_results.probability_quality": "Brier calibration requires independent labels.",
            "acceptance_gate_results.decision_benefit": "Typed cost must beat matched controls.",
            "acceptance_gate_results.retention": "Delayed improvement must survive restart.",
            "acceptance_gate_results.efficiency": "Whole service cost is the denominator.",
            "duration_s/phase_spans": "Measured monotonic time records actual work.",
            "random_seed/reproducibility_checksum": "Bind code, inputs, config and planned seeds.",
            "source_artifact_hashes/preconditions_checked": "Current and historical evidence retain distinct roles.",
            "validation_receipts/repository_health": "A new diagnostic cannot erase an old required failure.",
            "verifier_is_oracle/claim_scope": "Exact-label conformance is circular, not science.",
            "inference_substrate": "Duration rules follow actual work.",
            "MODEL_SPECS/model_invocation_counts": "No LLM was called by this task.",
            "contract_ready_score": "Qualified authority permits compute, never scientific benefit.",
            "authority_snapshot_paths/method_map_path": "Immutable rollover preserves earlier authority.",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "model_loads_attempted": 0,
            "generation_calls_attempted": 0,
            "forward_calls_attempted": 0,
            "tokens": 0,
            "model_file_hashes": [],
        },
        "contract_ready_score": 0,
        "authority_snapshot_paths": snapshots,
        "method_map_path": {
            "path": str(METHOD.relative_to(ROOT)),
            "sha256": sha256_file(METHOD) if METHOD.is_file() else None,
        },
        "contract_comparison": comparison,
        "historical_required_failures": json.loads(MANIFEST.read_text())[
            "historical_required_failures"
        ],
        "e2e_applicability": {
            "cli_e2e": "required real private entrypoint",
            "E2E-001_to_004": "no model training or native binding change",
            "E2E-007": "no certified bank update",
            "E2E-009_to_013": "no ARC runtime change",
        },
    }
    return value


def main(argv: list[str] | None = None) -> int:
    """Run the exact frozen roster and publish one terminal result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.raw is None:
        args.raw = args.output.parent / "rows.json" if args.no_validation else RAW / "rows.json"
    progress("start", "flushed_start", 0)
    if args.cold_replay is not None:
        passed = cold_replay(args.cold_replay, args.raw)
        progress("cold_replay", "passed" if passed else "failed", int(passed))
        return 0 if passed else 1
    if args.no_validation and args.output == RESULT:
        parser.error("--no-validation requires a private --output")
    RAW.mkdir(parents=True, exist_ok=True)
    if sha256_file(MANIFEST).removeprefix("sha256:") != MANIFEST_SHA256:
        raise ValueError("frozen validation manifest changed")
    manifest = json.loads(MANIFEST.read_text())
    source_rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path) if path.is_file() else None,
            "role": "historical_diagnosis" if "7823_v680" in str(path) else "current_contract",
            "date": args.date,
            "eligible": path.is_file(),
        }
        for path in REQUIRED_SOURCES
    ]
    missing = [path for path in REQUIRED_SOURCES if not path.is_file()]
    progress("preconditions", "checked", len(source_rows))
    if missing:
        comparison: dict[str, Any] = {
            "passed": False,
            "errors": ["missing_source"],
            "rows": [
                {"unit_id": f"exp{i}-unstarted", "status": "unstarted", "excluded": False}
                for i in range(7837, 7851)
            ],
        }
        snapshots: list[dict[str, str]] = []
    else:
        snapshots = snapshot_authorities((DESIGN, STAGED, ACTIVE), SNAPSHOTS)
        if not verify_snapshots(snapshots):
            raise ValueError("authority snapshot cold verification failed")
        comparison = compare_contract(
            DESIGN.read_text(),
            yaml.safe_load(STAGED.read_text()),
            yaml.safe_load(ACTIVE.read_text()),
        )
    progress("authority", "compared", 14)
    value = base_artifact(args.date, comparison, source_rows, snapshots, missing)
    atomic_json(args.raw, {"rows": value["rows"]})
    value["raw_rows_path"] = (
        str(args.raw.relative_to(ROOT)) if args.raw.is_relative_to(ROOT) else str(args.raw)
    )
    value["raw_rows_sha256"] = sha256_file(args.raw)
    value["duration_s"] = time.monotonic() - START
    value["phase_spans"] = {"preconditions_and_authority_s": value["duration_s"]}
    if missing:
        value["verdict_class"] = "blocked"
        value["honest_verdict"] = "complete_blocked_v681_missing_external_authority"
        atomic_json(args.output, value)
        return 0
    if not comparison["passed"]:
        value["verdict_class"] = "disqualified"
        value["honest_verdict"] = "complete_disqualified_v681_contract_mismatch"
        atomic_json(args.output, value)
        return 1
    if args.no_validation:
        atomic_json(args.output, value)
        progress("private_cli", "candidate_written", 14)
        return 0
    candidate = RAW / "candidate.json"
    atomic_json(candidate, value)
    receipts = []
    progress("validation", "start", 0)
    for index, command in enumerate(manifest["commands"]):
        receipts.append(run_child(command, index))
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [
        {
            "name": receipt["name"],
            "argv": receipt["command_argv"],
            "classification": receipt["classification"],
        }
        for receipt in receipts
    ]
    expected = [
        (command["name"], command["argv"], command["classification"])
        for command in manifest["commands"]
    ]
    observed = [
        (row["name"], row["argv"], row["classification"])
        for row in value["observed_child_commands"]
    ]
    all_pass = expected == observed and all(receipt["passed"] for receipt in receipts)
    all_pass = all_pass and verify_snapshots(snapshots) and cold_replay(candidate, args.raw)
    adversarial = next(receipt for receipt in receipts if receipt["name"] == "adversarial_verify")
    try:
        report = json.loads((ROOT / adversarial["log_path"]).read_text())
        flagged = bool(report["flagged_count"])
    except (OSError, ValueError, KeyError):
        flagged = True
    value["flagged_adversarial"] = flagged
    all_pass = all_pass and not flagged
    progress("repository_health", "before_subprocess", len(receipts))
    health = run_child(manifest["repository_health"], len(receipts))
    value["repository_health"] = health
    value["acceptance_gate_results"]["validity"] = all_pass
    value["acceptance_gate_results"]["readiness"] = int(all_pass)
    value["contract_ready_score"] = int(all_pass)
    value["verdict_class"] = "circular_positive" if all_pass else "disqualified"
    value["honest_verdict"] = (
        "complete_circular_positive_v681_contract_methods"
        if all_pass
        else "complete_disqualified_v681_contract_validation"
    )
    value["duration_s"] = time.monotonic() - START
    value["phase_spans"]["validation_and_health_s"] = (
        value["duration_s"] - value["phase_spans"]["preconditions_and_authority_s"]
    )
    atomic_json(args.output, value)
    progress("terminal", "artifact_written", len(receipts) + 1)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
