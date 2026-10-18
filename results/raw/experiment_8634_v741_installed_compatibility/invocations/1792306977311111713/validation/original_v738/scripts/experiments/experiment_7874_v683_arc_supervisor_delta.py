#!/usr/bin/env python3
"""Measure the post-V682 live supervisor receipt delta without game execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting.arc_supervisor_v683_delta import check_inputs, summarize
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

if __package__:
    from .experiment_7860_v682_arc_supervisor_delta import (
        _artifact as prior_artifact,
        _commands as prior_commands,
        cold_replay,
        validate,
    )
else:
    from experiment_7860_v682_arc_supervisor_delta import (
        _artifact as prior_artifact,
        _commands as prior_commands,
        cold_replay,
        validate,
    )

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "results/experiment_7860_v682_arc_supervisor_delta.json"
SOURCE = ROOT / "scripts/experiments/experiment_7860_v682_arc_supervisor_delta.py"
REGISTRY = ROOT / "ops/arc_solve_registry.yaml"
DELIVERABLE = ROOT / "results/experiment_7874_v683_arc_supervisor_delta.json"
EXPECTED = (
    "sha256:02339c224e2e9c2bfce03d2da7bbc136e810a2b0f42a609c9a2c60a1e549bf68",
    "sha256:7e64814e631350118462b108fc947ebac845a25eff339387a85a507b2ed3424d",
    "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
)


def progress(started: float, phase: str, units: int) -> None:
    """Report each boundary so a quiet reader is not mistaken for a hung job."""

    print(
        f"[exp7874] phase={phase} elapsed_s={time.monotonic() - started:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def commands(private: Path, cutoff: int) -> list[dict[str, Any]]:
    """Freeze exact child arguments before reading any candidate outcome."""

    old = prior_commands(private)
    replacements = {
        "test_arc_supervisor_receipt_delta_7860": "test_arc_supervisor_delta_7874",
        "arc_supervisor_receipt_delta": "arc_supervisor_v683_delta",
        "experiment_7860_v682_arc_supervisor_delta": "experiment_7874_v683_arc_supervisor_delta",
    }
    updated = json.dumps(old)
    for before, after in replacements.items():
        updated = updated.replace(before, after)
    fixed: list[dict[str, Any]] = json.loads(updated)
    py = str(ROOT / ".venv/bin/python")
    fixed.extend(
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
                "classification": "required",
                "deadline_s": 600,
            },
        ]
    )
    return fixed


def main() -> int:
    """Publish only after the sealed input and required child exits are known."""

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
        result = summarize(args.reduce_ledger, args.cutoff_ns, set(), {})
        atomic_json(args.output, result)
        progress(started, "private_delta_written", result["new_live_outcome_count"])
        return 0

    private = Path(tempfile.mkdtemp(prefix="carnot-exp7874-", dir="/tmp"))
    progress(started, "before_preconditions", 0)
    checked = check_inputs(PRIOR, SOURCE, REGISTRY, EXPECTED)
    progress(started, "after_preconditions", len(checked["checks"]))
    frozen = commands(private, checked["cutoff_ns"])
    manifest = private / "validation_command_manifest.json"
    cutoff_path = private / "cutoff_manifest.json"
    source_paths = (
        PRIOR,
        SOURCE,
        REGISTRY,
        ROOT / "python/carnot/reporting/arc_supervisor_v683_delta.py",
        ROOT / "tests/python/test_arc_supervisor_delta_7874.py",
        Path(__file__).resolve(),
    )
    source_hashes = {str(path): sha256_file(path) for path in source_paths if path.is_file()}
    atomic_json(
        manifest,
        {
            "commands": frozen,
            "source_hashes": source_hashes,
            "inapplicable_e2e": "Live game checks require a runtime change; this is a receipt reader.",
        },
    )
    atomic_json(
        cutoff_path,
        {
            "cutoff_ns": checked["cutoff_ns"],
            "prior_artifact_sha256": EXPECTED[0],
            "prior_source_sha256": EXPECTED[1],
            "registry_sha256": EXPECTED[2],
        },
    )
    progress(started, "frozen_manifest", len(frozen))
    delta = summarize(
        ROOT if not checked["failures"] else private,
        checked["cutoff_ns"],
        checked["prior_hashes"],
        checked["registry_levels"],
    )
    progress(started, "after_reduce", delta["new_live_outcome_count"])
    artifact = prior_artifact(
        args.date,
        started,
        checked["checks"],
        checked["failures"],
        cutoff_path,
        manifest,
        delta,
    )
    artifact.update(
        experiment_id=7874,
        task_id="exp7874-arc-supervisor-delta",
        milestone="2026.09.683",
        registry_cutoff=checked["cutoff_ns"],
        registry_sha256=EXPECTED[2],
        new_live_outcome_count=delta["new_live_outcome_count"],
        per_game_results=delta["per_game_results"],
        new_level_solves=delta["new_level_solves"],
        solve_provenance=None if not delta["new_level_solves"] else "live_agent_self_discovery",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        planned_inference_substrate_class="no_model_load",
        execution_venue="host",
        target_model="none (no pretrained model)",
        trained_head_specs=[],
        historical_model_identity="cited historical model identity only; no current model calls",
        source_artifact_hashes={
            **artifact["source_artifact_hashes"],
            **source_hashes,
            str(manifest): sha256_file(manifest),
            str(cutoff_path): sha256_file(cutoff_path),
        },
    )
    artifact["current_work_receipt"] = build_current_work_receipt(
        run_id=f"exp7874-{time.monotonic_ns()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_details={"new_live_outcome_count": delta["new_live_outcome_count"]},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=time.monotonic_ns() - int(artifact["duration_s"] * 1e9),
        ended_monotonic_ns=time.monotonic_ns(),
    )
    artifact["reproducibility_checksum"] = canonical_hash(
        {"sources": artifact["source_artifact_hashes"], "seed": 0, "cutoff": checked["cutoff_ns"]}
    )
    artifact["field_principles"].update(
        {
            "registry_cutoff": "The prior sealed snapshot fixes which outcomes are new.",
            "registry_sha256": "Exact registry bytes fix the already solved level boundary.",
            "new_live_outcome_count": "Only unique authenticated live receipts establish current N.",
            "per_game_results": "Game denominators and losses remain visible beside gains.",
        }
    )
    validate(artifact, private, frozen, started)
    candidate = private / "candidate.json"
    atomic_json(candidate, artifact)
    for name, argv in (
        (
            "terminal_adversarial",
            (
                str(ROOT / ".venv/bin/python"),
                "scripts/adversarial_verify.py",
                "--json",
                str(candidate),
            ),
        ),
        (
            "terminal_rows",
            (
                str(ROOT / ".venv/bin/python"),
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
        ),
    ):
        progress(started, f"before_{name}", 0)
        report = run_commands(
            ROOT, [CommandSpec(name, argv, "required", 60)], log_dir=private / name, heartbeat_s=30
        )[0]
        progress(started, f"after_{name}", int(report["passed"]))
        if not report["passed"]:
            artifact["flagged_adversarial"] = name == "terminal_adversarial"
            artifact["honest_verdict"] = "complete_disqualified_terminal_verification"
            artifact["verdict_class"] = "disqualified"
            artifact["arc_delta_ready_score"] = 0
            artifact["acceptance_gate_results"]["readiness"] = 0
            artifact["validation_errors"].append(name)
            atomic_json(candidate, artifact)
    atomic_json(args.output or DELIVERABLE, artifact)
    progress(started, "deliverable_written", delta["new_live_outcome_count"])
    return 0 if artifact["arc_delta_ready_score"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
