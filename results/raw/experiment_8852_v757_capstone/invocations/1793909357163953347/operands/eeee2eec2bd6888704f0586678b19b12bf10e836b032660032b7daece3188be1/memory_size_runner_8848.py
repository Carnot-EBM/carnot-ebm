"""REQ-REPORT-8848: budget and qualify a fixed sixty-window memory cost curve."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import memory_size_cost_8848 as m
from carnot.reporting.current_work_receipt import build_current_work_receipt
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def run(raw: Path, date: str) -> Json:
    """One scheduled block pilots all cells before admitting the remaining nine blocks."""
    began = time.monotonic_ns()
    m.progress("preconditions_before")
    pre = m.preflight(raw)
    checks = pre["checks"]
    resources = (
        checks["repository_virtualenv"]
        and checks["private_output"]
        and checks["free_disk_bytes"] >= checks["minimum_free_bytes"]
    )
    dependencies = [
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "python/carnot/memory/transactional_constraint_memory.py",
        "openspec/capabilities/research-reporting/spec.md",
    ]
    refs = [m.reference(m.ROOT / p) for p in m.OWNED + m.TESTS + dependencies]
    schedule = m.roster()
    m.durable(
        raw / "schedule.json",
        dict(
            rows=schedule,
            workloads=[m.workload(b) for b in range(10)],
            capacity=32,
            delay=8,
            pending_cap=32,
            query_count=64,
            restart_at=32,
            code_config_hashes=refs,
            total_budget_s=4200,
            validation_reserve_s=900,
            child_deadline_s=60,
            warmup_overlap="32 of64 requests repeat eight preparation structures; entity names overlap within a block; all answers are constructed support",
        ),
    )
    pre_end = time.monotonic_ns()
    m.progress("preconditions_after")
    rows, preparations, pilots = [], [], []
    projected, admitted = 0.0, not pre["failures"] and resources
    for block in range(10):
        block_preps = {}
        if admitted:
            for occupancy in (0, 8, 32):
                spec = dict(
                    block=block,
                    occupancy=occupancy,
                    arm="additions_only",
                    unit_id=f"prep-b{block}:n{occupancy}",
                    cluster_id=f"b{block}",
                    denominator=occupancy,
                )
                m.progress("preparation_before", len(preparations), 30 - len(preparations))
                prep = m.prepare(raw / f"prep-{block}-{occupancy}", spec, pre["head"])
                preparations.append(prep)
                block_preps[occupancy] = prep
                m.progress("preparation_after", len(preparations), 30 - len(preparations))
            admitted = all(m.preparation_parity(p) for p in block_preps.values())
        for spec in [r for r in schedule if r["block"] == block]:
            if admitted and (time.monotonic_ns() - began) / 1e9 < 3180:
                m.progress("window_before", len(rows), 60 - len(rows))
                row = m.window(
                    raw / f"window-{len(rows):02}",
                    spec,
                    pre["head"],
                    block_preps[spec["occupancy"]],
                )
                m.progress("window_after", len(rows) + 1, 59 - len(rows))
            else:
                row = dict(
                    spec,
                    status="censored",
                    missing_reason="external_admission"
                    if pre["failures"]
                    else "budget_or_pilot_failure",
                    metric=None,
                    complete_ns=None,
                    segments=[],
                    commands=[],
                    preparation=False,
                )
            rows.append(row)
        if block == 0:
            pilots = rows.copy()
            projected = (time.monotonic_ns() - pre_end) / 1e9 * 10 + 900
            admitted = admitted and all(m.parity(r)["passed"] for r in pilots) and projected < 4200
            m.progress(f"pilot_projection_seconds_{projected:.1f}_admitted_{admitted}", 6, 54)
    ended = time.monotonic_ns()
    work = dict(
        raw=str(raw.absolute()),
        run_date=date,
        preconditions=pre,
        resources_passed=bool(resources),
        schedule=schedule,
        window_rows=rows,
        preparation_cost_rows=preparations,
        pilot_window_rows=pilots,
        projection=dict(
            projected_total_s=projected,
            total_budget_s=4200,
            validation_reserve_s=900,
            admitted=admitted,
        ),
        code_refs=refs,
        owner_pid=os.getpid(),
        started_ns=began,
        ended_ns=ended,
        duration_s=(ended - began) / 1e9,
        phase_spans=[
            dict(phase="preconditions", started_ns=began, ended_ns=pre_end),
            dict(
                phase="preparation_pilot_and_scheduled_windows", started_ns=pre_end, ended_ns=ended
            ),
        ],
    )
    m.durable(raw / "measurement.json", work)
    return work


def build(work: Json, validation: Json) -> Json:
    """Qualify complete clocks independently of learning benefit or positive cost increments."""
    rows, preparations = work["window_rows"], work["preparation_cost_rows"]
    raw = Path(work["raw"])
    parity_rows = [m.parity(r) for r in rows]
    complete = sum(r["status"] == "completed" for r in rows)
    checks = validation["passed"] and validation["coverage"]["complete"]
    external = (
        bool(work["preconditions"]["failures"]) and not work["preconditions"]["internal_failure"]
    )
    execution = complete == 60 and len(preparations) == 30 and all(r["passed"] for r in parity_rows)
    ready = (
        checks and execution and not work["preconditions"]["failures"] and work["resources_passed"]
    )
    verdict = (
        "disqualified"
        if not checks or work["preconditions"]["internal_failure"] or (not external and not ready)
        else "blocked"
        if external
        else "null"
    )
    calls = sum(r["attempted_updates"] for r in rows + preparations if r["status"] == "completed")
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="cpu_numerical_training" if calls else "blocked_no_run",
        inference_substrate_details=dict(
            store="unchanged V754 Gibbs", numerical_gradient_calls=calls, llm_work=False
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["started_ns"],
        ended_monotonic_ns=work["ended_ns"],
        small_ebm_training=dict(performed=calls > 0, calls=calls),
    )
    refs = [
        m.reference(raw / name) for name in ("measurement.json", "schedule.json", "validation.json")
    ]
    refs += [
        m.reference(p)
        for p in sorted(raw.glob("*/*"))
        if p.is_file()
        and (
            p.name in {"request.json", "state.json", "window.json"}
            or p.name.startswith(("state-", "response-", "process-"))
        )
    ]
    refs += [
        dict(path=str(m.ROOT / r["log_path"]), sha256=r["log_sha256"])
        for r in validation["receipts"]
    ]
    if "combined_reference" in validation["coverage"]:
        refs.append(validation["coverage"]["combined_reference"])
    value = dict(
        experiment_id=8848,
        task_id=m.TASK,
        milestone="2026.11.757",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_memory_size_cost",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=calls,
        verifier_is_oracle=True,
        exposure_scope="constructed support worlds; warmup overlap32/64 on eight structures; entity names shared within block",
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
        phase_spans=work["phase_spans"],
        preconditions_checked=work["preconditions"],
        source_artifact_hashes=work["preconditions"]["sources"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=refs,
        rows=rows,
        intended_count=60,
        completed_count=complete,
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len({r["cluster_id"] for r in rows if r["status"] == "completed"}),
        gate_check_summary=work["preconditions"]["failures"],
        acceptance_gates={
            k: dict(passed=bool(v), principle=p)
            for k, v, p in [
                (
                    "complete_windows",
                    execution,
                    "Every scheduled window remains in the denominator.",
                ),
                ("owned_validation", checks, "Failed checks cannot qualify cost readiness."),
                (
                    "historical_authority",
                    not work["preconditions"]["failures"],
                    "New objectives cannot gate the admitted store.",
                ),
            ]
        },
        required_checks_passed=bool(checks),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        whole_cost_ready_score=int(ready),
        window_rows=rows,
        preparation_cost_rows=preparations,
        state_parity_rows=parity_rows,
        restart_rows=[
            dict(
                unit_id=r["unit_id"],
                matched=r["segments"][1]["restart_input_hash"] == r["segments"][0]["state_hash"],
            )
            for r in rows
            if r["status"] == "completed"
        ],
        serialization_bytes=[
            dict(
                unit_id=r["unit_id"],
                state_bytes=r["serialized_state_bytes"],
                response_bytes=r["response_bytes"],
            )
            for r in rows
            if r["status"] == "completed"
        ],
        unsupported_attribution=[
            "mode differences are whole-store interventions, not arithmetic costs",
            "no FPGA fractions",
            "no measured speedup or independent generalization",
        ],
        current_work_receipt=receipt,
        resource_projection=work["projection"],
        pilot_window_rows=work["pilot_window_rows"],
        complete_operation_boundary="Parent before first child startup/import through64 requests, restart at32, delay8, full tail drain, response serialization, store file/directory fsync and final child exit. Fixture installation precedes timing. No component clocks.",
        fsync_count_scope="Observed store barriers only; each child also durably writes snapshot and response outside this counter, inside whole wall clock.",
        primitive_reference=m.reference(raw / "measurement.json"),
        validation_reference=m.reference(raw / "validation.json"),
        flagged_adversarial=verdict == "disqualified",
        methodology_note="Fixed constructed five-predicate workloads on the unchanged V754 store. Ten paired blocks; no benefit claim. Preparation costs remain separate.",
        repository_health=validation.get("repository_health"),
        **m.reduce_costs(rows, preparations),
    )
    value["field_principles"] = {
        k: "Bind complete observed work to immutable operands; costs do not establish benefit."
        for k in value
    }
    return normalize_artifact_for_template_write(value)


def validate(raw: Path) -> Json:
    """Combine actual unit and child statements with zero exclusions; health stays separate."""
    config, data = raw / "coverage.ini", raw / ".coverage"
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
        CommandSpec(
            "owned_pytest",
            (
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
            ),
            "owned",
            240,
        ),
        CommandSpec(
            "private_E2E_018_021",
            (
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
            ),
            "owned",
            300,
        ),
        CommandSpec(
            "historical_store_controls",
            (
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
            ),
            "owned",
            180,
        ),
        CommandSpec("ruff_check", (str(m.ROOT / ".venv/bin/ruff"), "check", *paths), "owned", 60),
        CommandSpec(
            "ruff_format",
            (str(m.ROOT / ".venv/bin/ruff"), "format", "--check", *paths),
            "owned",
            60,
        ),
        CommandSpec(
            "strict_mypy",
            (str(m.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *m.OWNED),
            "owned",
            120,
        ),
        CommandSpec("spec_coverage", (py, "scripts/check_spec_coverage.py", *m.TESTS), "owned", 60),
    ]
    receipts = run_commands(m.ROOT, plan, log_dir=raw / "validation", heartbeat_s=60)
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
        files=files,
        zero_exclusions=not any(f["excluded"] for f in files),
        combined_reference=m.reference(data),
    )
    health = m.ROOT / "results/raw" / m.NAME / "global-health/full_python_once.receipt.json"
    result = dict(
        receipts=receipts,
        coverage=coverage,
        passed=all(r["passed"] for r in receipts) and coverage["complete"],
        repository_health=m.reference(health) if health.is_file() else None,
    )
    m.durable(raw / "validation.json", result)
    return result


def terminal(candidate: Path, raw: Path) -> Json:
    """Check prospective bytes with unchanged validators and a foreign-directory replay."""
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
                "sh",
                "-c",
                'cd /tmp && exec "$@"',
                "replay",
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
    """Publish only terminal bytes checked against their exact retained operands."""
    raw = Path(work["raw"])
    m.progress("owned_validation_before")
    validation = validate(raw)
    m.progress("owned_validation_after")
    value = build(work, validation)
    m.durable(raw / "prepublication.json", value)
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    m.durable(raw / "publication.json", publication)
    m.progress("publication_complete", 1, 0)
    return value


def main(argv: list[str] | None = None) -> int:
    """Use one producer for scheduled execution, private output and cold replay."""
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
