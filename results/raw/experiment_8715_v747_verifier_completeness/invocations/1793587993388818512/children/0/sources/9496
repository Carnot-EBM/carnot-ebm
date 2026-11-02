#!/usr/bin/env python3
"""Bind V682 authority and methods; REQ-REPORT-7851-V682."""

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
    build_current_work_receipt,
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
DESIGN = ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
ACTIVE = ROOT / "research-roadmap.yaml"
STAGED = ROOT / "research-roadmap-next.yaml"
METHOD = ROOT / "docs/research-notes/v682-method-map.md"
SNAPSHOTS = ROOT / "docs/research-notes/v682-authority-snapshots"
MANIFEST = ROOT / "docs/research-notes/v682-validation-manifest.json"
RESULT = ROOT / "results/experiment_7851_v682_contract_methods.json"
SEALED = ROOT / "docs/research-notes/v682-validation-logs"
START = time.monotonic()
START_NS = time.monotonic_ns()
SOURCES = (
    DESIGN,
    ACTIVE,
    METHOD,
    ROOT / "research-references.md",
    ROOT / "research-studying.md",
    ROOT / "research-complete.yaml",
    ROOT / "ops/exclusion_manifest.yaml",
    ROOT / "results/experiment_7837_v681_contract_methods.json",
    ROOT / "python/carnot/reporting/roadmap_contract.py",
    ROOT / "python/carnot/reporting/current_work_receipt.py",
    ROOT / "python/carnot/reporting/experiment_7303_validation_scope.py",
    ROOT / "tests/python/test_experiment_7851_v682_contract_methods.py",
    ROOT / "tests/python/test_experiment_7837_v681_contract_methods.py",
    Path(__file__),
)


def path_label(path: Path) -> str:
    """Keep worktree paths short and private fixture paths absolute."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def progress(phase: str, event: str, units: int) -> None:
    """Flush a monotonic boundary while keeping a count of finished units."""
    print(
        f"[exp7851] phase={phase} event={event} elapsed_s={time.monotonic() - START:.3f} completed_units={units}",
        flush=True,
    )


def run_child(command: dict[str, Any], index: int, scratch_dir: Path) -> dict[str, Any]:
    """Supervise one owned child and seal its log after process exit."""
    name = str(command["name"])
    argv = [str(part) for part in command["argv"]]
    deadline = float(command["timeout_s"])
    progress(name, "before_subprocess", index)
    started = time.monotonic()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH="python:.")
    with tempfile.NamedTemporaryFile(dir=scratch_dir, suffix=".log", delete=False) as stream:
        scratch = Path(stream.name)
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        timed_out = False
        while process.poll() is None:
            remaining = deadline - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            try:
                process.wait(timeout=min(30.0, remaining))
            except subprocess.TimeoutExpired:
                progress(name, "heartbeat", index)
        exit_code = process.wait()
    digest = sha256_file(scratch)
    target = SEALED / f"{name}-{digest.removeprefix('sha256:')}.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) != digest:
        raise ValueError("sealed log hash collision")
    if not target.exists():
        shutil.copyfile(scratch, target)
    output = scratch.read_text(errors="replace")
    scratch.unlink()
    try:
        log_label = str(target.relative_to(ROOT))
    except ValueError:
        log_label = str(target)
    receipt: dict[str, Any] = {
        "name": name,
        "command_argv": argv,
        "classification": command["classification"],
        "timeout_s": deadline,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": time.monotonic() - started,
        "log_path": log_label,
        "log_sha256": digest,
        "passed": exit_code == 0 and not timed_out,
        "output_tail": output[-2000:],
    }
    if name == "worktree_imports":
        try:
            imports = json.loads(output.strip().splitlines()[-1])["resolved_imports"]
        except (IndexError, ValueError, KeyError):
            imports = {}
        receipt["resolved_imports"] = imports
        receipt["passed"] = (
            receipt["passed"]
            and bool(imports)
            and all(Path(path).is_relative_to(ROOT / "python") for path in imports.values())
        )
    progress(name, f"after_subprocess_exit_{exit_code}", index + 1)
    return receipt


def source_rows(paths: tuple[Path, ...], date: str) -> list[dict[str, Any]]:
    """Record each exact byte source, including absent external evidence."""
    rows = []
    for path in paths:
        rows.append(
            {
                "path": path_label(path),
                "sha256": sha256_file(path) if path.is_file() else None,
                "date": date,
                "source_role": (
                    "historical_disqualified" if "7837_v681" in str(path) else "current_contract"
                ),
                "exposure_status": "historical" if "7837_v681" in str(path) else "current",
                "eligible": path.is_file(),
            }
        )
    return rows


def build_artifact(
    date: str,
    comparison: dict[str, Any],
    sources: list[dict[str, Any]],
    snapshots: list[dict[str, str]],
    missing: list[Path],
    roadmap_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build a pending receipt before running any validators."""
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
    count = len(rows)
    current = build_current_work_receipt(
        run_id=f"exp7851-{date}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"authority": str(roadmap_path.relative_to(ROOT))},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=START_NS,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    return {
        "experiment_id": 7851,
        "task_id": "exp7851-contract-methods",
        "milestone": "2026.09.682",
        "run_date": date,
        "honest_verdict": "partial_v682_validation_pending",
        "verdict_class": "partial",
        "flagged_adversarial": False,
        "gate_check_summary": gate_failures,
        "rows": rows,
        "sample_size_budget": {
            "intended": 14,
            "eligible": 0 if missing else count,
            "started": count if not missing else 0,
            "completed": sum(row.get("status") == "completed" for row in rows)
            if not missing
            else 0,
            "censored": 0,
            "excluded": sum(bool(row.get("excluded")) for row in rows),
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
        "duration_s": time.monotonic() - START,
        "phase_spans": {"preconditions_and_authority_s": time.monotonic() - START},
        "random_seed": None,
        "reproducibility_checksum": canonical_hash(
            {"sources": sources, "configuration": manifest, "seeds": [68201]}
        ),
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "source_paths": sources,
            "resource_ownership": "host_cpu_only_no_gpu_lease",
            "gate_operands": [],
            "roadmap_path": str(roadmap_path.relative_to(ROOT)),
            "schema_version": manifest.get("schema"),
            "missing": [str(path) for path in missing],
            "administrative_readiness_is_not_science_gate": True,
        },
        "validation_receipts": [],
        "validation_command_manifest_path": path_label(MANIFEST),
        "validation_command_manifest_sha256": sha256_file(MANIFEST) if MANIFEST.is_file() else None,
        "observed_child_commands": [],
        "repository_health": None,
        "verifier_is_oracle": True,
        "claim_scope": {
            "kind": "administrative_exact_authority",
            "science": "unmeasured",
            "natural_human_labels": "exposed_development_only",
            "fresh_generalization_eligible": False,
        },
        "field_principles": {
            "experiment_id/task_id/milestone/run_date": "Current producer identity is distinct from legacy aliases.",
            "honest_verdict/verdict_class": "The terminal prefix and class must agree.",
            "flagged_adversarial": "A verifier flag closes the gate.",
            "gate_check_summary": "Missing and wrong-valued operands have different causes.",
            "rows/sample_size_budget": "Primitive rows preserve every unit and independent count.",
            "acceptance_gate_results.validity": "Current checks must pass before use.",
            "acceptance_gate_results.readiness": "Exact contract agreement is administrative.",
            "acceptance_gate_results.probability_quality": "Calibration needs independent labels.",
            "acceptance_gate_results.decision_benefit": "Benefit needs a matched control.",
            "acceptance_gate_results.retention": "Learning needs delayed retained decisions.",
            "acceptance_gate_results.efficiency": "Whole service cost is the denominator.",
            "duration_s/phase_spans": "Monotonic spans record actual work.",
            "random_seed/reproducibility_checksum": "Bind source, configuration and seed bytes.",
            "source_artifact_hashes/preconditions_checked": "Historical and current sources differ in authority.",
            "validation_receipts/validation_command_manifest_path/observed_child_commands/repository_health": "A new wrapper cannot erase a failed required check.",
            "verifier_is_oracle/claim_scope": "Fixture agreement is circular evidence.",
            "inference_substrate/inference_substrate_class": "The runtime floor follows actual work.",
            "MODEL_SPECS/model_specs/model_invocation_counts": "This administrative task makes no model call.",
            "contract_ready_score": "Agreement has no scientific benefit claim.",
            "authority_snapshot_paths/method_map_path": "Immutable source bytes survive rollover.",
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
        "current_work_receipt": current,
        "contract_ready_score": 0,
        "authority_snapshot_paths": snapshots,
        "method_map_path": {
            "path": str(METHOD.relative_to(ROOT)),
            "sha256": sha256_file(METHOD) if METHOD.is_file() else None,
        },
        "contract_comparison": comparison,
        "historical_required_failures": manifest.get("historical_required_failures", []),
        "e2e_applicability": {
            "private_cli_and_cold_replay": "required",
            "E2E-001_to_004": "no training or native binding change",
            "E2E-007": "no bank update",
            "E2E-009_to_013": "no ARC runtime change",
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Execute the frozen V682 contract check and publish a terminal result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    progress("start", "flushed_start", 0)
    scratch = Path(os.environ.get("CARNOT_7851_TMP", "/tmp/carnot-exp7851-owned"))
    scratch.mkdir(parents=True, exist_ok=True)
    if args.cold_replay is not None:
        if args.raw is None:
            parser.error("--cold-replay requires --raw")
        passed = cold_replay(args.cold_replay, args.raw, experiment_id=7851)
        progress("cold_replay", "passed" if passed else "failed", int(passed))
        return 0 if passed else 1
    if args.no_validation and args.output == RESULT:
        parser.error("--no-validation requires a private --output")
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    staged = yaml.safe_load(STAGED.read_text()) if STAGED.is_file() else None
    roadmap_path = STAGED if staged and staged.get("milestone") == "2026.09.682" else ACTIVE
    paths = (*SOURCES, roadmap_path, MANIFEST)
    sources = source_rows(paths, args.date)
    missing = [path for path in paths if not path.is_file()]
    progress("preconditions", "checked", len(paths))
    if missing:
        comparison: dict[str, Any] = {
            "passed": False,
            "errors": ["missing_source"],
            "rows": [
                {
                    "unit_id": f"exp{i}-unstarted",
                    "arm": "four_source_contract",
                    "family": f"exp{i}",
                    "seed": None,
                    "status": "unstarted",
                    "censored": False,
                    "excluded": False,
                    "absolute_metric": 0,
                    "raw_numerator": 0,
                    "raw_denominator": 0,
                    "checks": {},
                }
                for i in range(7851, 7865)
            ],
        }
        snapshots: list[dict[str, str]] = []
    else:
        snapshots = snapshot_authorities((DESIGN, roadmap_path, ACTIVE), SNAPSHOTS)
        if not verify_snapshots(snapshots):
            raise ValueError("authority snapshot cold verification failed")
        comparison = compare_contract(
            DESIGN.read_text(),
            yaml.safe_load(roadmap_path.read_text()),
            yaml.safe_load(ACTIVE.read_text()),
            milestone="2026.09.682",
            first_id=7851,
            count=14,
        )
    progress("authority", "compared", 14 if not missing else 0)
    value = build_artifact(
        args.date, comparison, sources, snapshots, missing, roadmap_path, manifest
    )
    source_key = canonical_hash(
        {"sources": sources, "configuration": manifest, "seed": 68201}
    ).removeprefix("sha256:")
    raw = args.raw or scratch / f"rows-{source_key}.json"
    if raw.exists() and json.loads(raw.read_text()) != {"rows": value["rows"]}:
        raise ValueError("checkpoint content differs for same input hash")
    atomic_json(raw, {"rows": value["rows"]})
    value["raw_rows_path"] = str(raw)
    value["raw_rows_sha256"] = sha256_file(raw)
    if missing:
        value["verdict_class"] = "blocked"
        value["honest_verdict"] = "complete_blocked_v682_missing_external_authority"
        value["duration_s"] = time.monotonic() - START
        atomic_json(args.output, value)
        progress("terminal", "blocked_written", 0)
        return 0
    if not comparison["passed"]:
        value["verdict_class"] = "disqualified"
        value["honest_verdict"] = "complete_disqualified_v682_contract_mismatch"
        value["duration_s"] = time.monotonic() - START
        atomic_json(args.output, value)
        progress("terminal", "mismatch_written", 14)
        return 1
    if args.no_validation:
        value["duration_s"] = time.monotonic() - START
        atomic_json(args.output, value)
        progress("private_cli", "candidate_written", 14)
        return 0
    frozen = manifest.get("frozen_hashes", {})
    if any(sha256_file(ROOT / path) != digest for path, digest in frozen.items()):
        raise ValueError("frozen source or test changed")
    candidate = scratch / "pending-candidate.json"
    atomic_json(candidate, value)
    receipts = []
    progress("validation", "start", 0)
    for index, command in enumerate(manifest["commands"]):
        receipts.append(run_child(command, index, scratch))
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [
        {"name": row["name"], "argv": row["command_argv"], "classification": row["classification"]}
        for row in receipts
    ]
    expected = [(row["name"], row["argv"], row["classification"]) for row in manifest["commands"]]
    observed = [(row["name"], row["command_argv"], row["classification"]) for row in receipts]
    passed = expected == observed and all(row["passed"] for row in receipts)
    passed = (
        passed and verify_snapshots(snapshots) and cold_replay(candidate, raw, experiment_id=7851)
    )
    adversarial = next(row for row in receipts if row["name"] == "adversarial_verify")
    try:
        report = json.loads((ROOT / adversarial["log_path"]).read_text())
        flagged = bool(report["flagged_count"])
    except (OSError, ValueError, KeyError):
        flagged = True
    value["flagged_adversarial"] = flagged
    passed = passed and not flagged
    health = run_child(manifest["repository_health"], len(receipts), scratch)
    value["repository_health"] = health
    value["acceptance_gate_results"]["validity"] = passed
    value["acceptance_gate_results"]["readiness"] = int(passed)
    value["contract_ready_score"] = int(passed)
    value["verdict_class"] = "circular_positive" if passed else "disqualified"
    value["honest_verdict"] = (
        "complete_circular_positive_v682_contract_methods"
        if passed
        else "complete_disqualified_v682_contract_validation"
    )
    value["duration_s"] = time.monotonic() - START
    value["phase_spans"]["validation_and_health_s"] = (
        value["duration_s"] - value["phase_spans"]["preconditions_and_authority_s"]
    )
    atomic_json(args.output, value)
    progress("terminal", "artifact_written", len(receipts) + 1)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
