"""REQ-REPORT-8835: budgeted execution and publication of complete operation costs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import complete_adaptation_cost_8835 as m
from carnot.reporting.current_work_receipt import build_current_work_receipt
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]


def run(raw: Path, date: str) -> Json:
    """Two fixed pilots must leave enough time for the full schedule and validation."""
    began = time.monotonic_ns()
    m.progress("preconditions_before")
    pre = m.preflight(raw)
    checks = pre["checks"]
    resources = (
        checks["repository_virtualenv"]
        and checks["private_output"]
        and checks["free_disk_bytes"] >= checks["minimum_free_bytes"]
    )
    code_refs = [m.reference(m.ROOT / p) for p in m.OWNED + m.TESTS]
    code_refs += [
        m.reference(m.ROOT / p)
        for p in [
            m.STORE,
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/experiment_7303_validation_scope.py",
            "scripts/experiment_template.py",
            "python/carnot/memory/transactional_constraint_memory.py",
            "python/carnot/models/gibbs/__init__.py",
            "python/carnot/training/nce.py",
            "python/carnot/reporting/evidence_set_methods_8768.py",
        ]
    ]
    schedule = m.roster()
    m.durable(
        raw / "schedule.json",
        dict(
            rows=schedule,
            code_config_hashes=code_refs,
            seed=m.SEED,
            arm_admission_rows=pre["arm_admission_rows"],
            component_timers=False,
            total_budget_s=4200,
            validation_reserve_s=900,
            child_deadline_s=60,
        ),
    )
    pre_end = time.monotonic_ns()
    m.progress("preconditions_after")
    pilots = []
    if not pre["failures"] and resources:
        for index, arm in enumerate(("read_only", "combined")):
            spec = dict(schedule[index], unit_id=f"pilot-{index}:{arm}", block=-1, arm=arm)
            m.progress("pilot_before", index, 2 - index)
            pilots.append(m.window(raw / f"pilot-{index}", spec, pre["head"]))
            m.progress("pilot_after", index + 1, 1 - index)
    projected = max((r["complete_ns"] / 1e9 for r in pilots), default=0) * 90
    admitted = (
        len(pilots) == 2
        and all(m.parity(r)["passed"] for r in pilots)
        and (time.monotonic_ns() - began) / 1e9 + projected + 900 < 4200
    )
    rows = []
    for index, spec in enumerate(schedule):
        if admitted and (time.monotonic_ns() - began) / 1e9 < 4200 - 900 - 120:
            m.progress("window_before", index, 90 - index)
            row = m.window(raw / f"window-{index:03}", spec, pre["head"])
            m.progress("window_after", index + 1, 89 - index)
        else:
            row = dict(
                spec,
                status="censored",
                missing_reason="external_admission"
                if pre["failures"]
                else "budget_or_pilot_failure",
                metric=None,
                complete_ns=None,
                service_ns=None,
                segments=[],
                commands=[],
            )
        rows.append(row)
    ended = time.monotonic_ns()
    work = dict(
        raw=str(raw.absolute()),
        run_date=date,
        preconditions=pre,
        resources_passed=bool(resources),
        schedule=schedule,
        window_rows=rows,
        pilot_window_rows=pilots,
        projection=dict(
            projected_execution_s=projected, validation_reserve_s=900, admitted=admitted
        ),
        code_refs=code_refs,
        phase_spans=[
            dict(phase="preconditions", started_ns=began, ended_ns=pre_end),
            dict(phase="pilots_and_full_schedule", started_ns=pre_end, ended_ns=ended),
        ],
        started_ns=began,
        ended_ns=ended,
        duration_s=(ended - began) / 1e9,
        owner_pid=os.getpid(),
    )
    m.durable(raw / "measurement.json", work)
    return work


def build(work: Json, validation: Json) -> Json:
    """Readiness means complete valid timings, independently of scientific benefit."""
    rows = work["window_rows"]
    raw = Path(work["raw"])
    parity_rows = [m.parity(r) for r in rows]
    complete = sum(r["status"] == "completed" for r in rows)
    internal = work["preconditions"]["internal_failure"]
    external = bool(work["preconditions"]["failures"]) and not internal
    checks_passed = validation["passed"] and validation["coverage"]["complete"]
    execution = complete == 90 and all(r["passed"] for r in parity_rows)
    ready = checks_passed and execution and not external and work["resources_passed"]
    verdict = (
        "disqualified"
        if internal or not checks_passed or (not external and not ready)
        else "blocked"
        if external
        else "null"
    )
    windows = rows + work["pilot_window_rows"]
    calls = sum(len(s["releases"]) for r in windows for s in r["segments"])
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="cpu_numerical_training",
        inference_substrate_details=dict(
            store="shipped V754 Gibbs mixed loss", llm_work=False, numerical_gradient_calls=calls
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["started_ns"],
        ended_monotonic_ns=work["ended_ns"],
        small_ebm_training=dict(performed=calls > 0, calls=calls),
    )
    refs = [
        m.reference(raw / "measurement.json"),
        m.reference(raw / "schedule.json"),
        m.reference(raw / "validation.json"),
    ]
    refs += [m.reference(p) for p in sorted(raw.glob("*/response-*.json"))]
    refs += [m.reference(p) for p in sorted(raw.glob("*/state-*.json"))]
    refs += [m.reference(p) for p in sorted(raw.glob("*/process-*.log"))]
    refs += [m.reference(p) for p in sorted(raw.glob("*/request.json"))]
    if "combined_reference" in validation["coverage"]:
        refs.append(validation["coverage"]["combined_reference"])
    for r in validation["receipts"]:
        refs.append(dict(path=str(m.ROOT / r["log_path"]), sha256=r["log_sha256"]))
    health = m.ROOT / "results/raw" / m.NAME / "global-health/full_python_once.receipt.json"
    value = dict(
        repository_health_reference=m.reference(health) if health.is_file() else None,
        experiment_id=8835,
        task_id=m.TASK,
        milestone="2026.11.756",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_complete_adaptation_cost",
        verdict_class=verdict,
        inference_substrate="cpu_numerical_training",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=calls,
        numerical_training_scope="90 scheduled windows plus two fixed pilots; validation work has separate logs; unchanged read-only gradients are attempted but not committed",
        verifier_is_oracle=True,
        exposure_scope="constructed_exact_five_predicate_streams",
        independent_generalization_score=0,
        duration_s=work["duration_s"],
        random_seed=m.SEED,
        reproducibility_checksum=m.canonical_hash(
            dict(
                schedule=work["schedule"],
                sources=work["preconditions"]["sources"],
                code=work["code_refs"],
            )
        ),
        source_artifact_hashes=work["preconditions"]["sources"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=refs,
        phase_spans=work["phase_spans"],
        preconditions_checked=work["preconditions"],
        rows=rows,
        intended_count=90,
        completed_count=complete,
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len({r["cluster_id"] for r in rows if r["status"] == "completed"}),
        gate_check_summary=work["preconditions"]["failures"],
        acceptance_gates={
            name: dict(passed=bool(passed), principle=principle)
            for name, passed, principle in [
                (
                    "immutable_historical_authority",
                    not external,
                    "Historical cost evidence does not depend on a new objective.",
                ),
                (
                    "complete_valid_windows",
                    execution,
                    "All90 windows and exact state controls must pass.",
                ),
                (
                    "owned_validation",
                    checks_passed,
                    "Actual tests and100-percent new statements control readiness.",
                ),
                ("scientific_speed_gain", False, "Benefit is not required for cost readiness."),
            ]
        },
        required_checks_passed=bool(checks_passed),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        whole_cost_ready_score=int(ready),
        window_rows=rows,
        parent_span_rows=[
            dict(
                unit_id=r["unit_id"],
                commands=r["commands"],
                complete_ns=r["complete_ns"],
                service_ns=r["service_ns"],
            )
            for r in rows
        ],
        state_parity_rows=parity_rows,
        arm_admission_rows=work["preconditions"]["arm_admission_rows"],
        complete_operation_boundary=m.BOUNDARY,
        current_work_receipt=receipt,
        pilot_window_rows=work["pilot_window_rows"],
        resource_projection=work["projection"],
        primitive_reference=m.reference(raw / "measurement.json"),
        validation_reference=m.reference(raw / "validation.json"),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        flagged_adversarial=verdict == "disqualified",
        methodology_note="Parent-only complete-operation timing on the exact shipped causal store. Matched mode differences include unchanged observer and gradient work. Thirty reset stream blocks describe constructed CPU costs, not natural-text generalization. No component fractions, native implementation or measured100x speedup.",
        **m.reduce_costs(rows),
    )
    value["field_principles"] = {
        k: "Bind actual scheduled work and exact raw operands; complete costs establish no learning or hardware speedup."
        for k in value
    }
    return value


def validate(raw: Path) -> Json:
    """Combine real unit and subprocess statements with no coverage exclusions."""
    config = raw / "coverage.ini"
    data = raw / ".coverage"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ndata_file = "
        + str(data)
        + "\ninclude =\n"
        + "".join("    " + str(m.ROOT / p) + "\n" for p in m.OWNED)
        + "\n[report]\nexclude_lines =\n"
    )
    py = str(m.ROOT / ".venv/bin/python")
    paths = m.OWNED + m.TESTS
    plan = [
        (
            "owned_pytest",
            [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *m.TESTS,
            ],
            240,
        ),
        (
            "private_E2E_018_021",
            [
                py,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_restricted_decision_audit_8210.py",
            ],
            300,
        ),
        (
            "historical_V754_controls",
            [
                py,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_causal_memory_8802.py",
            ],
            180,
        ),
        ("ruff_check", [str(m.ROOT / ".venv/bin/ruff"), "check", *paths], 60),
        ("ruff_format", [str(m.ROOT / ".venv/bin/ruff"), "format", "--check", *paths], 60),
        (
            "strict_mypy",
            [str(m.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *m.OWNED],
            120,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", *m.TESTS], 60),
    ]
    receipts = run_commands(
        m.ROOT,
        [CommandSpec(n, tuple(a), "owned", t) for n, a, t in plan],
        log_dir=raw / "validation",
        heartbeat_s=60,
    )
    cov = Coverage(data_file=str(data), config_file=str(config))
    cov.combine(keep=True)
    cov.save()
    files = []
    for name in m.OWNED:
        _, statements, excluded, missing, _ = cov.analysis2(str(m.ROOT / name))
        files.append(
            dict(path=name, statements=len(statements), excluded=excluded, missing=missing)
        )
    statements = sum(f["statements"] for f in files)
    missing_count = sum(len(f["missing"]) for f in files)
    coverage = dict(
        complete=statements > 0 and missing_count == 0 and not any(f["excluded"] for f in files),
        percent=100 * (statements - missing_count) / max(1, statements),
        zero_exclusions=not any(f["excluded"] for f in files),
        files=files,
        combined_reference=m.reference(data),
    )
    result = dict(
        receipts=receipts,
        coverage=coverage,
        passed=all(r["passed"] for r in receipts) and coverage["complete"],
    )
    m.durable(raw / "validation.json", result)
    return result


def terminal(candidate: Path, raw: Path) -> Json:
    """Use unchanged consumers and an actual outside-checkout replay process."""
    py = str(m.ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(candidate), "--json"),
            "owned",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "owned",
            60,
        ),
        CommandSpec(
            "cold_replay",
            (
                "env",
                "-u",
                "PYTHONPATH",
                "-C",
                "/tmp",
                py,
                "-u",
                str(m.ROOT / m.CLI),
                "--replay",
                str(candidate),
            ),
            "owned",
            60,
        ),
    ]
    receipts = run_commands(m.ROOT, commands, log_dir=raw / "terminal", heartbeat_s=60)
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def qualify(work: Json, output: Path) -> Json:
    """Publication exposes only a candidate checked against its exact retained operands."""
    raw = Path(work["raw"])
    m.progress("owned_validation_before")
    validation = validate(raw)
    m.progress("owned_validation_after")
    value = build(work, validation)
    m.durable(raw / "prepublication.json", value)
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    m.durable(raw / "publication.json", publication)
    report = json.loads(Path(publication["sidecar_path"]).read_text())["report"]
    m.durable(raw / "terminal_validation.json", dict(report, publication=publication))
    m.progress("publication_complete", 1, 0)
    return value


def main(argv: list[str] | None = None) -> int:
    """Private output and replay share the same producer; no historical artifact is rewritten."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--start", type=int, choices=[0, 32], default=0)
    args = parser.parse_args(argv)
    if args.worker:
        m.worker(args.worker, args.start)
        return 0
    if args.replay:
        report = m.replay(args.replay)
        print(json.dumps(report), flush=True)
        return int(not report["passed"])
    raw = (
        args.raw or m.ROOT / "results/raw" / m.NAME / "invocations" / str(time.time_ns())
    ).absolute()
    qualify(run(raw, args.date), args.output.absolute())
    return 0
