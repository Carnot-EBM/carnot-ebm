"""REQ-REPORT-8862: qualify complete linear costs independently of learning benefit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import linear_durable_cost_8862 as m
from carnot.reporting import memory_size_runner_8848 as base
from carnot.reporting.current_work_receipt import build_current_work_receipt
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def run(raw: Path, date: str) -> Json:
    """Pilot one full paired block and budget all sixty windows plus validation."""
    began = time.monotonic_ns()
    m.progress("preconditions_before")
    pre = m.preflight(raw)
    resources = (
        all(pre["checks"][k] for k in ("repository_virtualenv", "private_output"))
        and pre["checks"]["free_disk_bytes"] >= pre["checks"]["minimum_free_bytes"]
    )
    schedule = m.roster()
    sources = raw / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    refs = []
    for path in m.OWNED + m.TESTS:
        source = m.ROOT / path
        target = sources / source.name
        shutil.copyfile(source, target)
        refs += [m.reference(source), m.reference(target)]
    m.durable(
        raw / "schedule.json",
        dict(
            rows=schedule,
            workloads=[m.workload(b) for b in range(10)],
            pending_cap=32,
            template_cap=32,
            delay=8,
            restart_at=24,
            queries=48,
            total_budget_s=4200,
            validation_reserve_s=900,
            child_deadline_s=60,
            parameter_motion="L2 steps are observations, never timing shares",
        ),
    )
    pre_end = time.monotonic_ns()
    m.progress("preconditions_after", 1)
    rows, preparations = [], []
    admitted = not pre["failures"] and resources
    censor_reason = (
        "external_" + pre["failures"][0]["observed"]
        if pre["failures"]
        else "private_resource_precondition_failed"
    )
    projected = 0.0
    numerical: Json = {}
    for block in range(10):
        prepared = {}
        if admitted:
            for occupancy in (0, 32):
                spec = dict(
                    schedule[block * 6],
                    occupancy=occupancy,
                    arm="linear_additions_only",
                    unit_id=f"prep-b{block}:n{occupancy}",
                    denominator=occupancy,
                )
                m.progress("preparation_before", len(preparations), 20 - len(preparations))
                prep = m.execute(
                    raw / f"prep-{block}-{occupancy}",
                    spec,
                    pre["heads"][str(spec["seed"])],
                    preparation=True,
                )
                preparations.append(prep)
                prepared[occupancy] = prep
                m.progress("preparation_after", len(preparations), 20 - len(preparations))
            admitted = all(m.parity(r)["passed"] for r in prepared.values())
            if not admitted:
                censor_reason = "preparation_control_failed"
        for spec in schedule[block * 6 : (block + 1) * 6]:
            if admitted and (time.monotonic_ns() - began) / 1e9 < 3180:
                m.progress("window_before", len(rows), 60 - len(rows))
                row = m.execute(
                    raw / f"window-{len(rows):02}",
                    spec,
                    pre["heads"][str(spec["seed"])],
                    prepared=prepared[spec["occupancy"]],
                )
                m.progress("window_after", len(rows) + 1, 59 - len(rows))
            else:
                row = dict(
                    spec,
                    status="censored",
                    missing_reason=censor_reason
                    if not admitted
                    else "measurement_budget_reserving_validation",
                    metric=None,
                    complete_ns=None,
                    segments=[],
                    commands=[],
                    preparation=False,
                )
            rows.append(row)
        if block == 0:
            projected = (time.monotonic_ns() - pre_end) / 1e9 * 10 + 900
            if admitted:
                censor_reason = (
                    "pilot_projection_exceeds_4200_seconds"
                    if projected >= 4200
                    else "pilot_state_parity_failed"
                )
            admitted = admitted and all(m.parity(r)["passed"] for r in rows) and projected < 4200
            m.progress(f"pilot_projection_s_{projected:.1f}_admitted_{admitted}", 6, 54)
    if admitted:
        numerical = m.microbenchmark(pre["heads"][str(schedule[0]["seed"])])
    ended = time.monotonic_ns()
    work = dict(
        raw=str(raw.absolute()),
        run_date=date,
        preconditions=pre,
        resources_passed=bool(resources),
        schedule=schedule,
        window_rows=rows,
        preparation_cost_rows=preparations,
        microbenchmark=numerical,
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
            dict(phase="preparation_and_windows", started_ns=pre_end, ended_ns=ended),
        ],
    )
    m.durable(raw / "measurement.json", work)
    return work


def build(work: Json, validation: Json) -> Json:
    """Grant cost readiness only after all complete transactions and owned checks pass."""
    rows, preps = work["window_rows"], work["preparation_cost_rows"]
    raw = Path(work["raw"])
    parity = [m.parity(r) for r in rows + preps]
    complete = sum(r["status"] == "completed" for r in rows)
    checks = validation["passed"] and validation["coverage"]["complete"]
    external = (
        bool(work["preconditions"]["failures"]) and not work["preconditions"]["internal_failure"]
    )
    execution = complete == 60 and len(preps) == 20 and all(r["passed"] for r in parity)
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
    calls = sum(r.get("applied_updates", 0) for r in rows + preps) + work["microbenchmark"].get(
        "numerical_training_calls", 0
    )
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="cpu_numerical_training" if calls else "blocked_no_run",
        inference_substrate_details=dict(backend="JAX CPU linear SGD", llm_work=False),
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
    reduced = m.reduce_costs(rows, preps)
    value = dict(
        experiment_id=8862,
        task_id=m.TASK,
        milestone="2026.11.758",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_linear_durable_cost",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=calls,
        verifier_is_oracle=True,
        exposure_scope="constructed support worlds;24/48 repeat prepared structures; no natural-language claim",
        independent_generalization_score=0,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        preconditions_checked=work["preconditions"],
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
                    "complete_windows_and_preparation",
                    execution,
                    "Every paired durable window and actual admission must complete.",
                ),
                ("owned_validation", checks, "Failed checks suppress readiness."),
                (
                    "historical_authority",
                    not work["preconditions"]["failures"],
                    "Only authenticated original methods are admitted.",
                ),
            ]
        },
        required_checks_passed=bool(checks),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        whole_cost_ready_score=int(ready),
        window_rows=rows,
        preparation_cost_rows=preps,
        restart_rows=[
            dict(
                unit_id=r["unit_id"],
                matched=r["segments"][1]["restart_input_hash"] == r["segments"][0]["state_hash"],
            )
            for r in rows
            if r["status"] == "completed"
        ],
        state_parity_rows=parity,
        movement_rows=[
            dict(
                unit_id=r["unit_id"],
                parameter_movement=r["parameter_movement"],
                gradient_norm_sum=r["gradient_norm_sum"],
                attempted_updates=r["attempted_updates"],
                applied_updates=r["applied_updates"],
            )
            for r in rows
            if r["status"] == "completed"
        ],
        serialization_bytes=[
            dict(
                unit_id=r["unit_id"],
                store_bytes=r["serialized_bytes"],
                snapshot_bytes=r["serialized_state_bytes"],
                response_bytes=r["response_bytes"],
            )
            for r in rows + preps
            if r["status"] == "completed"
        ],
        unsupported_attribution=[
            "No numerical batch subtraction or service component shares.",
            "No native or FPGA numerical-only result.",
            "No demonstrated service speed improvement or independent generalization.",
        ],
        numerical_microbenchmark=work["microbenchmark"],
        current_work_receipt=receipt,
        resource_projection=work["projection"],
        complete_operation_boundary="Parent before startup/imports through48 queries, restart24, delay8, tail drain, state acknowledgements, serialization, file/directory fsync and final child exit. Prepared fixture installation precedes timing.",
        fsync_count_scope="Actual store barriers; snapshot/response/log barriers also occur inside parent time but outside store counters.",
        primitive_reference=m.reference(raw / "measurement.json"),
        validation_reference=m.reference(raw / "validation.json"),
        flagged_adversarial=verdict == "disqualified",
        methodology_note="Ten paired independent blocks; two occupancies; three admitted linear arms. Whole-service costs and separately timed preparation; motion is not timing attribution.",
        repository_health=validation.get("repository_health"),
        **reduced,
    )
    value["field_principles"] = {
        k: "Bind actual complete work to retained operands; cost readiness does not establish scientific benefit."
        for k in value
    }
    return normalize_artifact_for_template_write(value)


def validate(raw: Path) -> Json:
    """Reuse the frozen validator with only this producer's owned paths replaced."""
    with patch.multiple(base.m, OWNED=m.OWNED, TESTS=m.TESTS, NAME=m.NAME, CLI=m.CLI):
        return base.validate(raw)


def terminal(candidate: Path, raw: Path) -> Json:
    """Apply unchanged scanners and a foreign-directory replay to prospective bytes."""
    with patch.object(base.m, "CLI", m.CLI):
        return base.terminal(candidate, raw)


def qualify(work: Json, output: Path) -> Json:
    """Publish only the exact candidate that passes terminal checks."""
    raw = Path(work["raw"])
    m.progress("owned_validation_before")
    validation = validate(raw)
    m.progress("owned_validation_after", 1)
    value = build(work, validation)
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    m.durable(raw / "publication.json", publication)
    m.progress("publication_complete", 1)
    return value


def main(argv: list[str] | None = None) -> int:
    """Keep private storage options separate from the frozen scientific census."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261106"], default="20261106")
    parser.add_argument("--raw", "--raw-dir", dest="raw", type=Path)
    parser.add_argument(
        "--output", "--private-output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json")
    )
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--start", type=int, choices=[0, 24], default=0)
    args = parser.parse_args(argv)
    if args.worker:
        m.worker(args.worker, args.start)
        return 0
    if args.replay:
        report = m.replay(args.replay)
        print(json.dumps(report), flush=True)
        return int(not report["passed"])
    raw = (
        args.raw or args.output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    ).absolute()
    qualify(run(raw, args.date), args.output.absolute())
    return 0
