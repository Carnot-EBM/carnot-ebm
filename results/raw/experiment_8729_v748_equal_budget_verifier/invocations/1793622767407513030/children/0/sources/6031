"""Own bounded validation and publication for REQ-VERIFY-7930-V688."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.publication_qualification_7928 import terminal_checks
from carnot.verify import energy_fit_7930 as core

TESTS = ("tests/python/test_energy_fit_7930.py", "tests/python/test_energy_fit_7930_run.py")
CONSUMERS = (
    "tests/python/test_energy_fit_7894.py",
    "tests/python/test_natural_runtime_7867.py",
    "tests/python/test_natural_runtime_7853.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_conductor_gates.py",
    "tests/python/test_in_process_doc_reconcile.py",
    "tests/python/test_current_work_receipt.py",
)
INCLUDES = ",".join("*/" + name for name in core.OWNED)


def freeze(private: Path, raw: Path) -> tuple[Path, list[CommandSpec]]:
    """Keep exact commands and identical coverage scope fixed before head fitting."""
    py = str(core.ROOT / ".venv/bin/python")
    pytest = str(core.ROOT / ".venv/bin/pytest")
    cli = core.OWNED[-1]
    cov = (py, "-m", "coverage")
    measure = (
        *cov,
        "run",
        "--parallel-mode",
        f"--data-file={private / '.coverage'}",
        f"--include={INCLUDES}",
    )
    common = ("-n", "0", "-o", "addopts=", "--no-cov", "-q")
    commands = [
        CommandSpec(
            "affected_unit",
            (
                *measure,
                "-m",
                "pytest",
                *common,
                f"--basetemp={private / 'unit'}",
                *TESTS,
                *CONSUMERS,
            ),
            "required",
            600,
        ),
        CommandSpec(
            "e2e_015",
            (
                pytest,
                *common,
                f"--basetemp={private / 'e2e015'}",
                "tests/python/test_source_boundary_7852.py",
            ),
            "required",
            180,
        ),
        CommandSpec(
            "e2e_016_fixture",
            (
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--fixture-e2e",
                str(private / "e2e016.json"),
            ),
            "required",
            180,
        ),
        CommandSpec(
            "e2e_016_replay",
            (
                py,
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--cold-replay",
                str(private / "e2e016.json"),
            ),
            "required",
            180,
        ),
        CommandSpec(
            "cli_expected_failure",
            (
                *measure,
                cli,
                "--date",
                "20260930",
                "--runtime",
                str(private / "missing.json"),
                "--output",
                str(private / "experiment_7930_negative.json"),
                "--assert-ready",
            ),
            "expected_failure",
            90,
        ),
        CommandSpec(
            "cli_success",
            (
                *measure,
                cli,
                "--date",
                "20260930",
                "--runtime",
                str(private / "missing.json"),
                "--output",
                str(private / "experiment_7930_blocked.json"),
            ),
            "required",
            90,
        ),
        CommandSpec(
            "cli_blocked_replay",
            (
                *measure,
                cli,
                "--date",
                "20260930",
                "--cold-replay",
                str(private / "experiment_7930_blocked.json"),
            ),
            "required",
            90,
        ),
        CommandSpec(
            "ruff_check",
            (str(core.ROOT / ".venv/bin/ruff"), "check", *core.OWNED, *TESTS),
            "required",
            120,
        ),
        CommandSpec(
            "ruff_format",
            (str(core.ROOT / ".venv/bin/ruff"), "format", "--check", *core.OWNED, *TESTS),
            "required",
            120,
        ),
        CommandSpec(
            "mypy", (str(core.ROOT / ".venv/bin/mypy"), "--strict", *core.OWNED), "required", 180
        ),
        CommandSpec(
            "spec_coverage",
            (py, "scripts/check_spec_coverage.py", *TESTS, *CONSUMERS),
            "required",
            180,
        ),
        CommandSpec("full_pytest", (pytest, "tests/python", "-q"), "repository_health", 600),
        CommandSpec(
            "cli_cold_replay",
            (*measure, cli, "--date", "20260930", "--cold-replay", str(raw / "candidate.json")),
            "late",
            600,
        ),
        CommandSpec(
            "coverage_combine",
            (*cov, "combine", f"--data-file={private / '.coverage'}", str(private)),
            "late",
            30,
        ),
        CommandSpec(
            "coverage_json",
            (
                *cov,
                "json",
                f"--data-file={private / '.coverage'}",
                f"--include={INCLUDES}",
                "--fail-under=100",
                "-o",
                str(private / "coverage.json"),
            ),
            "late",
            30,
        ),
    ]
    path = raw / "validation_command_manifest.json"
    atomic_json(
        path,
        {
            "commands": [
                {
                    **asdict(c),
                    "expected_exit": 2 if c.scope == "expected_failure" else 0,
                    "expected_reason": "source evidence blocked"
                    if c.scope == "expected_failure"
                    else "command must pass",
                    "deadline_s": c.timeout_s,
                }
                for c in commands
            ],
            "coverage_includes": INCLUDES,
            "dependencies": {
                name: sha256_file(core.ROOT / name)
                for name in (
                    *core.OWNED,
                    "scripts/experiments/experiment_7894_v685_energy_fit.py",
                    "python/carnot/reporting/primary_publication.py",
                    "python/carnot/reporting/publication_qualification_7928.py",
                    "scripts/conductor_gates.py",
                    "scripts/in_process_doc_reconcile.py",
                )
            },
            "execution_date": "20260930",
            "historical_fixture_date": "20260929",
            "configuration": {
                "arms": core.ARMS,
                "seeds": core.SEEDS,
                "epochs": 16,
                "learning_rate": 0.01,
                "width": 16,
                "parameter_limit": 4096,
                "base_features": 132,
                "complete_static_features": 16,
                "compute_budget_s": 3000,
                "kl_tolerance": 0.01,
                "alternate_ce_limit": 0.70,
                "dual_step": 0.01,
                "dual_clip": [0, 10],
                "temperatures": core.training_runtime.temperature_grid(),
            },
        },
    )
    return path, commands


def execute(commands: list[CommandSpec], raw: Path) -> list[dict[str, Any]]:
    """Inspect real exited child logs, retaining expected failures as passing checks."""
    private_coverage = Path(tempfile.mkdtemp(prefix="carnot7930-child-coverage-")) / ".coverage"
    rows = run_commands(
        core.ROOT,
        commands,
        log_dir=raw / "logs",
        heartbeat_s=30,
        extra_env={"COVERAGE_FILE": str(private_coverage)},
    )
    for spec, row in zip(commands, rows, strict=True):
        row.update(
            {
                "argv": list(spec.argv),
                "actual_exit": row["exit_code"],
                "expected_exit": 2 if spec.scope == "expected_failure" else 0,
                "deadline_s": spec.timeout_s,
                "measured_files": list(core.OWNED),
            }
        )
        if spec.scope == "expected_failure":
            row["expected_reason"] = "source evidence blocked"
            row["passed"] = (
                row["exit_code"] == 2
                and "source evidence blocked" in Path(row["log_path"]).read_text()
            )
    return rows


def base(
    q: Any, upstream: Path, sources: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> dict[str, Any]:
    """Producer identity and empty compute fields remain honest on blocked paths."""
    result: dict[str, Any] = q.base(upstream, sources, failures)
    result.update(
        {
            "experiment_id": 7930,
            "task_id": "exp7930-energy-fit",
            "milestone": "2026.09.688",
            "run_date": "20260930",
            "execution_date": "20260930",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "inference_substrate_class": "blocked_no_run" if failures else "no_model_load",
            "claim_scope": "exposed_development; no independent generalization claim",
            "upstream_path": str(upstream),
            "prediction_rows_sha256": None,
            "coverage_statement_counts": {},
            "primary_resolution_receipt": {
                "binding": "external receipt records final reader-selected path and byte hash"
            },
            "terminal_validation_sidecar_path": None,
        }
    )
    return result


def publish(value: dict[str, Any], output: Path) -> None:
    """Check final bytes atomically, then authenticate both actual consumer selections."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"]["path"] = str(raw / "primary_resolution_receipt.json")
    value["duration_s"] = time.monotonic() - core.START
    for key in value:
        value["field_principles"].setdefault(
            key, "Bind current measured evidence to exact producer, input bytes and declared role."
        )
    staging = raw / "candidate.json"
    atomic_json(staging, value)
    report = terminal_checks(staging, raw)
    if not report["passed"] or report["flagged_adversarial"]:
        value.update(
            {
                "honest_verdict": "complete_disqualified_terminal_validation",
                "verdict_class": "disqualified",
                "flagged_adversarial": report["flagged_adversarial"],
                "energy_fit_ready_score": 0,
            }
        )
        value["acceptance_gate_results"].update({"validity": False, "readiness": 0})
    receipt = core.publication.publish_primary(output, value, lambda p: terminal_checks(p, raw))
    atomic_json(
        raw / "terminal_validation.json",
        {
            "candidate_sha256": receipt["primary_sha256"],
            "validator_sidecar_path": receipt["sidecar_path"],
        },
    )
    selected = core.publication.reader_receipt(
        "exp7930-energy-fit",
        output.parent,
        field="energy_fit_ready_score",
        expected=value["energy_fit_ready_score"],
    )
    if not selected["passed"] or selected["gate_sha256"] != receipt["primary_sha256"]:
        raise ValueError("primary consumer mismatch")
    atomic_json(raw / "primary_resolution_receipt.json", selected)
    core.progress(
        "publish", f"stable sha256={receipt['primary_sha256']}", len(value["trained_head_specs"])
    )


def enrich(
    value: dict[str, Any],
    prediction_path: Path,
    spans: list[dict[str, Any]],
    timings: dict[str, float],
    private: Path,
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce primitive response energies and measured batches without a benefit claim."""
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    for row in rows:
        p = min(1 - 1e-15, max(1e-15, row["raw_risk"]))
        row.update(
            {
                "energy_unsupported": float(-np.log(p)),
                "energy_supported": float(-np.log1p(-p)),
                "energy_scope": "effective normalized response energy",
                "measurement_batch_duration_s": timings[
                    f"arm={row['arm']} seed={row['seed']} role={row['role']}"
                ],
            }
        )
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    value["rows"], value["prediction_rows_sha256"] = rows, sha256_file(prediction_path)
    coverage_path = private / "coverage.json"
    coverage = (
        json.loads(coverage_path.read_text()).get("files", {}) if coverage_path.is_file() else {}
    )
    value["coverage_statement_counts"] = {name: item["summary"] for name, item in coverage.items()}
    covered = all(
        any(
            name.endswith(owned)
            and item["summary"]["num_statements"] > 0
            and item["summary"]["missing_lines"] == 0
            for name, item in coverage.items()
        )
        for owned in core.OWNED
    )
    failures = [r for r in receipts if not r["passed"] and r["scope"] != "repository_health"]
    ready = (
        covered
        and not failures
        and len(value["trained_head_specs"]) == 27
        and len(rows) == value["sample_size_budget"]["eligible"]
    )
    value.update(
        {
            "honest_verdict": "complete_null_energy_fit"
            if ready
            else "complete_disqualified_required_checks",
            "verdict_class": "null" if ready else "disqualified",
            "energy_fit_ready_score": int(ready),
            "phase_spans": spans,
            "validation_receipts": receipts,
            "observed_child_commands": [r["argv"] for r in receipts],
        }
    )
    value["acceptance_gate_results"].update(
        {"validity": ready, "readiness": int(ready), "calibration": None}
    )
    value["repository_health"].update(
        {
            "current_full_pytest": [r for r in receipts if r["scope"] == "repository_health"],
            "retire_if_same_verdict": True,
        }
    )
    value["sample_size_budget"].update(
        {"independent": 0, "exposed_development_families": len(rows) // 27}
    )
    for values in value["sample_size_budget"]["by_role"].values():
        values.update(
            {
                "started": values["eligible"] * 27,
                "completed": values["eligible"] * 27,
                "failed": 0,
                "censored": 0,
                "independent": 0,
            }
        )
    value["field_principles"].update(
        {
            "energy_fit_ready_score": "All 27 current heads, complete family rows and passing owned checks make a null reusable.",
            "coverage_statement_counts": "Nonempty exact added-code coverage is required; repository debt remains separate.",
            "shortcut_control_rows": "Fit and tune targets alone fit shortcuts; length-matched permutation tests source dependence.",
            "primary_resolution_receipt": "Actual gate and document readers must select the published bytes after newer nested sidecars.",
            "claim_scope": "Exposed development families cannot establish independent generalization.",
            "rows": "Primitive probabilities, normalized response energies and labels permit cold reduction.",
            "gate_check_summary": "External input failures name their exact path, hash, field, operator and observed value.",
        }
    )
    return value


def produce(upstream: Path, runtime: Path, output: Path) -> int:
    """Keep external blocks terminal and numerical deadlines separate from owned failures."""
    q = core.library(upstream, output)
    original_base = q.base
    from types import SimpleNamespace

    q.base = lambda u, s, f: base(SimpleNamespace(base=original_base), u, s, f)
    raw = output.parent / "raw" / output.stem
    private = Path(tempfile.mkdtemp(prefix="carnot7930-"))
    manifest_path, commands = freeze(private, raw)
    sources: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    batches: dict[str, float] = {}

    def progress(phase: str, event: str = "boundary", units: int = 0) -> None:
        if phase == "score":
            key = event.split(" ", 1)[1]
            if event.startswith("before"):
                batches[key] = time.monotonic()
            else:
                timings[key] = time.monotonic() - batches[key]
        core.progress(phase, event, units)

    q.progress = progress
    phase_start = time.monotonic() - core.START

    def end_phase(name: str) -> None:
        nonlocal phase_start
        end = time.monotonic() - core.START
        spans.append(
            {"phase": name, "start_s": phase_start, "end_s": end, "duration_s": end - phase_start}
        )
        phase_start = end
        core.progress(name, "complete", len(spans))

    try:
        core.progress("preconditions", "before exact primary and transitive hashes")
        references, dependencies, qualification = core.authenticate(runtime, upstream)
        public, excluded, sources = core.public_records(upstream)
        sources = [*references, *sources]
        frozen = json.loads(manifest_path.read_text())
        dependency_hash = canonical_hash(
            {
                "runtime": dependencies,
                "frozen": frozen["dependencies"],
                "config": frozen["configuration"],
                "inputs": sources,
            }
        )
        records = core.attach_labels(public, upstream, {"fit", "tune"})
        end_phase("preconditions")
        began = time.monotonic()
        checkpoints, checkpoint_path = core.fit_heads(records, raw, dependency_hash, 3000)
        end_phase("fit")
        if len(checkpoints) != 27:
            raise ValueError("all 27 current heads must be sealed before later-role labels")
        records = core.attach_labels(public, upstream, set(core.prior.ROLE_BUDGET))
        prediction_path, _ = q.score(records, checkpoints, raw)
        end_phase("score")
        control_rows = q.controls(records, checkpoints, raw)
        end_phase("controls")
        if time.monotonic() - began > 3000:
            raise TimeoutError("single numerical budget exhausted")
        receipts = execute([c for c in commands if c.scope != "late"], raw)
        end_phase("validation")
        value = q.candidate(
            records,
            excluded,
            sources,
            checkpoints,
            checkpoint_path,
            prediction_path,
            control_rows,
            manifest_path,
            receipts,
        )
        value["historical_required_failures"] += qualification["historical_required_failures"]
        value["preconditions_checked"].update(
            {
                "training_runtime_ready_score": 1,
                "current_dependency_hash": dependency_hash,
                "later_role_labels_accessed_after_sealed_heads": 27,
            }
        )
        value["training_dependency_hashes"] = dependencies
        value["resolved_imports"].update(
            {
                "carnot.verify.energy_fit_7930": str(Path(core.__file__).resolve()),
                "carnot.verify.energy_fit_7930_run": str(Path(__file__).resolve()),
            }
        )
        atomic_json(raw / "candidate.json", value)
        receipts += execute([c for c in commands if c.scope == "late"], raw)
        end_phase("cold_reconstruction_and_coverage")
        value = enrich(value, prediction_path, spans, timings, private, receipts)
        value["reproducibility_checksum"] = dependency_hash
    except core.custody.InputBlocked as exc:
        core.progress("preconditions", "source evidence blocked")
        value = q.base(upstream, sources, exc.operands)
        end_phase("external_block")
        value["phase_spans"] = spans
    except (TimeoutError, ValueError, KeyError) as exc:
        value = q.base(upstream, sources, [])
        partial = isinstance(exc, TimeoutError)
        value.update(
            {
                "honest_verdict": "complete_partial_numerical_work"
                if partial
                else "complete_disqualified_owned_work",
                "verdict_class": "partial" if partial else "disqualified",
                "unfinished_reason": str(exc),
                "checkpoint_manifest_path": str(raw / "checkpoint_manifest.json"),
                "phase_spans": spans,
            }
        )
        if partial:
            atomic_json(output, value)
            return 1
    value["validation_command_manifest_path"] = str(manifest_path)
    publish(value, output)
    return 0
