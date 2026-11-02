#!/usr/bin/env python3
"""Publish the current live supervisor receipt delta. REQ-REPORT-7899-V685."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting.arc_supervisor_v685_delta import reduce_receipts, replay_delta
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "results/experiment_7887_v684_arc_supervisor_delta.json"
REGISTRY = ROOT / "ops/arc_solve_registry.yaml"
AGENT = ROOT / "python/carnot/agentic/arc_competition_agent.py"
OUTPUT = ROOT / "results/experiment_7899_v685_arc_supervisor_delta.json"
PRIVATE = ROOT / "results/raw/experiment_7899_v685_arc_supervisor_delta"
EXPECTED = {
    PRIOR: "sha256:d0d952ae5bee679388852803fce0051e1dbbbb269e2de5cf7c8aa48794198af5",
    REGISTRY: "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
    AGENT: "sha256:0ab79da07e9aff18a270b5fe9a682ddef1a309c359405ff400cf4dd2174bf914",
}
MODULE = "python/carnot/reporting/arc_supervisor_v685_delta.py"
CLI = "scripts/experiments/experiment_7899_v685_arc_supervisor_delta.py"
TESTS = (
    "tests/python/test_arc_supervisor_delta_7899.py",
    "tests/python/test_arc_supervisor_delta_7887.py",
    "tests/python/test_arc_supervisor_delta_7874.py",
)


def progress(started: float, phase: str, units: int) -> None:
    """Show phase boundaries so a silent child cannot look complete."""

    print(
        f"[exp7899] phase={phase} elapsed_s={time.monotonic() - started:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def precheck() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], dict[str, int]]:
    """Check exact upstream bytes and retain prior identities despite its failed readiness."""

    checks = []
    for path, expected in EXPECTED.items():
        actual = sha256_file(path) if path.is_file() else "missing"
        checks.append(
            {
                "upstream_id": path.stem,
                "path": str(path),
                "sha256": None if actual == "missing" else actual,
                "artifact_field": "sha256",
                "op": "==",
                "expected": expected,
                "observed": actual,
                "role": "delta_baseline"
                if path == PRIOR
                else "registry_precheck"
                if path == REGISTRY
                else "live_policy_source",
                "exposure_status": "read_only",
            }
        )
    failures = [row for row in checks if row["expected"] != row["observed"]]
    baseline: dict[str, str] = {}
    levels: dict[str, int] = {}
    if not failures:
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        for row in prior.get("outcome_rows", []):
            if row.get("event_id") and row.get("content_sha256"):
                baseline[row["event_id"]] = row["content_sha256"]
            if row.get("source_sha256"):
                baseline["raw:" + row["source_sha256"]] = row["source_sha256"]
        parsed = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
        games = parsed.get("games", {})
        if isinstance(games, dict):
            levels = {
                str(name): int(value["levels_reproduced"])
                for name, value in games.items()
                if isinstance(value, dict) and "levels_reproduced" in value
            }
        elif isinstance(games, list):
            levels = {
                str(value["game"]): int(value["levels_reproduced"])
                for value in games
                if isinstance(value, dict) and "game" in value and "levels_reproduced" in value
            }
        checks.append(
            {
                "upstream_id": PRIOR.stem,
                "path": str(PRIOR),
                "sha256": sha256_file(PRIOR),
                "artifact_field": "verdict_class",
                "op": "==",
                "expected": "disqualified",
                "observed": prior.get("verdict_class"),
                "role": "historical_verdict",
                "exposure_status": "read_only",
            }
        )
        if checks[-1]["expected"] != checks[-1]["observed"]:
            failures.append(checks[-1])
    return checks, failures, baseline, levels


def commands(private: Path) -> list[dict[str, Any]]:
    """Freeze real paths, negative semantics and identical coverage includes."""

    py = str(ROOT / ".venv/bin/python")
    cov = str(ROOT / ".venv/bin/coverage")
    pytest = str(ROOT / ".venv/bin/pytest")
    include = f"*/{MODULE},*/{CLI}"
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    test_scratch = Path(tempfile.mkdtemp(prefix="carnot-exp7899-tests-", dir="/tmp"))
    fixture = private / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    atomic_json(
        fixture / "results/experiment_9001_arc.json",
        {
            "run_date": "20260929",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "source_artifact_hashes": {},
        },
    )
    producer = str(fixture / "results/experiment_9001_arc.json")
    base = [CLI, "--reduce-ledger", str(fixture), "--producer", producer]
    output = str(private / "fixture_delta.json")
    rows: list[dict[str, Any]] = []

    def add(
        name: str,
        argv: list[str],
        deadline: int = 120,
        expected_exit: int = 0,
        expected_text: str | None = None,
    ) -> None:
        rows.append(
            {
                "name": name,
                "argv": argv,
                "deadline_s": deadline,
                "classification": "required",
                "expected_exit": expected_exit,
                "expected_text": expected_text,
            }
        )

    add(
        "affected_pytest",
        [pytest, *TESTS, *common, f"--basetemp={test_scratch / 'test-temp'}"],
        240,
    )
    add(
        "unit_coverage",
        [
            cov,
            "run",
            f"--data-file={private / 'coverage.unit'}",
            f"--include={include}",
            "-m",
            "pytest",
            *TESTS,
            *common,
            f"--basetemp={test_scratch / 'coverage-temp'}",
        ],
        240,
    )
    add(
        "cli_success_coverage",
        [
            cov,
            "run",
            f"--data-file={private / 'coverage.success'}",
            f"--include={include}",
            *base,
            "--output",
            output,
        ],
    )
    add(
        "cli_failure_coverage",
        [cov, "run", f"--data-file={private / 'coverage.failure'}", f"--include={include}", *base],
        60,
        2,
        "--output",
    )
    add(
        "cli_replay_coverage",
        [
            cov,
            "run",
            f"--data-file={private / 'coverage.replay'}",
            f"--include={include}",
            CLI,
            "--cold-replay",
            output,
        ],
    )
    add(
        "coverage_combine",
        [
            cov,
            "combine",
            f"--data-file={private / 'coverage.combined'}",
            *(
                str(private / f"coverage.{name}")
                for name in ("unit", "success", "failure", "replay")
            ),
        ],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            f"--data-file={private / 'coverage.combined'}",
            f"--include={include}",
            "--show-missing",
            "--fail-under=100",
        ],
    )
    add("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", MODULE, CLI, *TESTS])
    add("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", MODULE, CLI, *TESTS])
    add("mypy", [str(ROOT / ".venv/bin/mypy"), "--strict", MODULE, CLI], 180)
    add("scoped_spec", [py, "scripts/check_spec_coverage.py", *TESTS], 120)
    add("e2e_017", [pytest, "tests/python/test_arc_supervisor_delta_7874.py", *common], 120)
    add(
        "e2e_016_fixture",
        [
            py,
            "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
            "--date",
            "20260929",
            "--fixture-e2e",
            str(private / "e2e016.json"),
        ],
        240,
    )
    add(
        "e2e_016_replay",
        [
            py,
            "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
            "--date",
            "20260929",
            "--cold-replay",
            str(private / "e2e016.json"),
        ],
        240,
    )
    return rows


def _run(row: dict[str, Any], private: Path, started: float) -> dict[str, Any]:
    """Run a bounded child, then seal its closed log at a durable hash path."""

    progress(started, "before_" + row["name"], 0)
    spec = CommandSpec(row["name"], tuple(row["argv"]), row["classification"], row["deadline_s"])
    receipt = run_commands(ROOT, [spec], log_dir=private / "logs" / row["name"], heartbeat_s=30)[0]
    original = ROOT / receipt["log_path"]
    digest = sha256_file(original).removeprefix("sha256:")
    sealed = private / "sealed" / f"{row['name']}_{digest}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if not sealed.exists():
        shutil.copyfile(original, sealed)
    receipt["log_path"] = str(sealed)
    receipt["log_sha256"] = "sha256:" + digest
    receipt["classification"] = row["classification"]
    receipt["expected_exit"] = row["expected_exit"]
    receipt["expected_text"] = row["expected_text"]
    receipt["passed"] = (
        receipt["exit_code"] == row["expected_exit"]
        and not receipt["timed_out"]
        and (row["expected_text"] is None or row["expected_text"] in receipt["output_tail"])
    )
    progress(started, "after_" + row["name"], int(receipt["passed"]))
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run private routes or publish one exact terminal candidate."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--reduce-ledger", type=Path)
    parser.add_argument("--producer", type=Path, action="append")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "start", 0)
    if args.cold_replay is not None:
        doc = json.loads(args.cold_replay.read_text(encoding="utf-8"))
        errors = replay_delta(doc)
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    if args.reduce_ledger is not None:
        if not args.producer or args.output is None:
            parser.error("--reduce-ledger requires --producer and --output")
        progress(started, "before_private_reduce", 0)
        delta = reduce_receipts(args.reduce_ledger, args.producer, {}, {})
        atomic_json(args.output, delta)
        progress(started, "after_private_reduce", delta["new_outcome_count"])
        return 0

    progress(started, "before_precheck", 0)
    checks, failures, baseline, registry = precheck()
    progress(started, "after_precheck", len(checks))
    private = PRIVATE
    private.mkdir(parents=True, exist_ok=True)
    manifest_rows = commands(private)
    manifest = private / "validation_command_manifest.json"
    atomic_json(
        manifest,
        {
            "commands": manifest_rows,
            "affected_closure": [MODULE, CLI, *TESTS],
            "applicable_e2e": ["E2E-016", "E2E-017"],
            "inapplicable_e2e": [f"E2E-{number:03d}" for number in range(1, 16)] + ["E2E-018"],
            "scope_rationale": "current ARC reader and direct CLI, V684 and V683 consumer regressions",
        },
    )
    producers = [
        path
        for path in sorted((ROOT / "results").glob("experiment_*arc*.json"))
        if path.name.split("_")[1].isdecimal()
        and int(path.name.split("_")[1]) > 7887
        and path != OUTPUT
    ]
    progress(started, "before_reduce", 0)
    delta = (
        reduce_receipts(ROOT, producers, baseline, registry)
        if not failures
        else reduce_receipts(ROOT, [], baseline, registry)
    )
    progress(started, "after_reduce", delta["new_outcome_count"])
    inventory = private / "receipt_inventory.json"
    atomic_json(
        inventory,
        {
            "baseline": baseline,
            "current": {
                row["event_id"]: row["content_sha256"]
                for row in delta["rows"]
                if row.get("event_id")
            },
            "producer_paths": [str(path) for path in producers],
        },
    )
    sources = {
        str(path): sha256_file(path)
        for path in (*EXPECTED, ROOT / MODULE, Path(__file__), manifest, inventory)
        if path.is_file()
    }
    sources.update(
        {
            row["source_path"]: row["source_sha256"]
            for row in delta["rows"]
            if row.get("source_sha256")
        }
    )
    prior = json.loads(PRIOR.read_text(encoding="utf-8")) if PRIOR.is_file() else {}
    artifact: dict[str, Any] = {
        "experiment_id": 7899,
        "task_id": "exp7899-arc-supervisor-delta",
        "milestone": "2026.09.685",
        "run_date": args.date,
        "honest_verdict": "complete_blocked_source_precondition"
        if failures
        else "complete_null_no_new_supervisor_outcomes",
        "verdict_class": "blocked" if failures else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": delta["rows"],
        "outcome_rows": delta["rows"],
        "sample_size_budget": delta["sample_size_budget"],
        "acceptance_gate_results": {
            "validity": None if failures else True,
            "readiness": 0 if failures else 1,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": [],
        "random_seed": 0,
        "reproducibility_checksum": canonical_hash(
            {"sources": sources, "baseline": baseline, "seed": 0}
        ),
        "source_artifact_hashes": sources,
        "preconditions_checked": checks,
        "resolved_imports": {"carnot.reporting.arc_supervisor_v685_delta": str(ROOT / MODULE)},
        "validation_receipts": [],
        "validation_command_manifest_path": str(manifest),
        "observed_child_commands": [],
        "historical_required_failures": prior.get("historical_required_failures", [])
        + [
            {
                "experiment_id": 7887,
                "honest_verdict": prior.get("honest_verdict"),
                "validation_errors": prior.get("validation_errors", []),
            }
        ],
        "repository_health": {
            "status": "historical_diagnostic_open",
            "repository_wide_check_repeated": False,
        },
        "verifier_is_oracle": False,
        "claim_scope": "exposed_development; observational live receipt delta",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none",
        "trained_head_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "file_hashes": []},
        "arc_delta_ready_score": 0 if failures else 1,
        "per_game_results": delta["per_game_results"],
        "receipt_inventory_path": str(inventory),
        "cutoff_receipt_hashes": baseline,
        "new_outcome_count": delta["new_outcome_count"],
        "new_live_outcome_count": delta["new_outcome_count"],
        "firings": delta["firings"],
        "new_level_solves": 0,
        "recommendation_rows": delta["recommendation_rows"],
        "registry_precheck": registry,
        "solve_provenance": [row.get("solve_provenance") for row in delta["rows"]],
        "validation_errors": [],
    }
    artifact["field_principles"] = {
        key: "Current producer bytes and primitive counts govern this field." for key in artifact
    }
    receipts: list[dict[str, Any]] = []
    if not failures:
        for row in manifest_rows:
            receipts.append(_run(row, private, started))
        artifact["validation_receipts"] = receipts
        artifact["observed_child_commands"] = [row["command_argv"] for row in receipts]
        artifact["validation_errors"] = [row["name"] for row in receipts if not row["passed"]]
        if artifact["validation_errors"]:
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["verdict_class"] = "disqualified"
            artifact["arc_delta_ready_score"] = 0
            artifact["acceptance_gate_results"].update(validity=False, readiness=0)
    artifact["duration_s"] = time.monotonic() - started
    artifact["phase_spans"] = [
        {"phase": "precheck_reduce_validate", "duration_s": artifact["duration_s"]}
    ]
    candidate = private / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    for attempt in range(2):
        reports = []
        for name, argv in (
            (
                "terminal_adversarial",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(candidate),
                ],
            ),
            (
                "terminal_rows",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
            ),
        ):
            reports.append(
                _run(
                    {
                        "name": name,
                        "argv": argv,
                        "deadline_s": 90,
                        "classification": "required",
                        "expected_exit": 0,
                        "expected_text": None,
                    },
                    private,
                    started,
                )
            )
        atomic_json(
            private / f"terminal_reports_{attempt}.json",
            {"candidate_sha256": sha256_file(candidate), "reports": reports},
        )
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
    atomic_json(target, json.loads(candidate.read_text(encoding="utf-8")))
    progress(started, "deliverable_written", delta["new_outcome_count"])
    return 0 if artifact["arc_delta_ready_score"] else 1


if __name__ == "__main__":
    sys.exit(main())
