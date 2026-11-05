"""REQ-REPORT-8847: freeze operands, execute paired processes and publish checked costs."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import indexed_cost_8847 as m
from carnot.reporting import indexed_witness_runner_8845 as previous
from carnot.reporting import ordered_evidence_runner_8831 as ordered
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.selective_methods_runner_8829 import qualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
UPSTREAM = m.ROOT / "results/experiment_8845_v757_indexed_witness.json"
PIN = "sha256:d99cfc2ce8e439be67d3a7d1713c7c403563b6dc59c3d2b1da8307a3b63789f8"
HEALTH = m.ROOT / "results/raw" / m.NAME / "repository_health/receipt.json"


def preconditions(raw: Path) -> Json:
    """Reuse preserved V756 authorities, then authenticate the opt-in index and its receipts."""
    pre = previous.preconditions(raw)
    diagnostic: Json = dict(
        check="missing_file",
        upstream="exp8845-indexed-witness",
        artifact_field="indexed_witness_ready_score",
        expected=1,
        observed=None,
        path=str(UPSTREAM),
        sha256=None,
    )
    try:
        ref = reference(UPSTREAM)
        value = load(ref)
        diagnostic.update(
            check="missing_field",
            observed=value.get("indexed_witness_ready_score"),
            sha256=ref["sha256"],
        )
        if "indexed_witness_ready_score" not in value:
            raise ValueError("missing_field")
        diagnostic["check"] = "observed_zero"
        if value["indexed_witness_ready_score"] != 1:
            raise ValueError("readiness_zero")
        diagnostic.update(
            check="rejected_producer", artifact_field="sha256", expected=PIN, observed=ref["sha256"]
        )
        if (
            ref["sha256"] != PIN
            or value["task_id"] != "exp8845-indexed-witness"
            or not value["required_checks_passed"]
        ):
            raise ValueError("producer_identity")
        pre["references"].append(ref)
        for operand in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            diagnostic.update(path=operand["path"], expected=operand["sha256"], observed=None)
            observed = sha256_file(Path(operand["path"]))
            diagnostic.update(observed=observed, sha256=observed)
            if observed != operand["sha256"]:
                raise ValueError("operand_drift")
            pre["references"].append(operand)
    except (OSError, ValueError, KeyError, TypeError) as error:
        pre["diagnostics"].append(
            dict(
                diagnostic,
                error=str(error),
                check="missing_file" if isinstance(error, OSError) else diagnostic["check"],
            )
        )
    pre["external_ready"] = not pre["diagnostics"]
    pre["evaluation_worlds"] = []
    if pre["external_ready"]:
        units = [
            u for u in load(reference(raw / "inputs.json"))["rows"] if u["split"] == "evaluation"
        ]
        pre["evaluation_worlds"] = sorted({u["cluster_id"] for u in units})
        if (
            len(units) != 1440
            or len(pre["evaluation_worlds"]) != 240
            or any(canonical_hash(u["input"]) != u["input_sha256"] for u in units)
        ):
            raise ValueError("evaluation_roster")
        atomic_json(
            raw / "frozen_inputs.json",
            dict(
                rows=units + m.stress(),
                cap=m.CAP,
                schedules=list(m.SCHEDULES),
                bootstrap_draws=m.DRAWS,
            ),
        )
    return dict(pre)


def process_measure(raw: Path, units: list[Json]) -> list[Json]:
    """Each scheduling seed runs both resident arms in separate, randomly ordered processes."""
    atomic_json(raw / "requests.json", dict(rows=units))
    rows = []
    for seed in m.SCHEDULES:
        arms = list(m.ARMS)
        random.Random(seed).shuffle(arms)
        for arm in arms:
            output = raw / f"{seed}_{arm}.json"
            name = f"measure_{seed}_{arm}"
            m.progress("before_" + name)
            receipt = stream(
                name,
                [
                    sys.executable,
                    "-u",
                    str(m.ROOT / m.CLI),
                    "--measure-arm",
                    arm,
                    "--schedule-seed",
                    str(seed),
                    "--inputs",
                    str(raw / "requests.json"),
                    "--decisions",
                    str(output),
                ],
                raw / "process_logs",
                deadline=300,
            )
            atomic_json(raw / f"{name}.receipt.json", receipt)
            m.progress("after_" + name)
            if not receipt["passed"]:
                raise ValueError("measurement_child_failure")
            rows.extend(load(reference(output))["rows"])
    return rows


def prepare(raw: Path, limit: int | None = None, measurement: bool = True) -> Json:
    """Durably seal raw predictions before opening encoded truth or computing any gain gate."""
    start = time.monotonic_ns()
    m.progress("before_preconditions")
    pre = preconditions(raw)
    m.progress("after_preconditions")
    rows, units = [], []
    if measurement and pre["external_ready"] and pre["resources"]["passed"]:
        units = load(reference(raw / "frozen_inputs.json"))["rows"][:limit]
        rows = process_measure(raw, units)
        atomic_json(
            raw / "sealed_predictions.json", dict(rows=rows, commitment=canonical_hash(rows))
        )
        m.progress("before_labels_and_bootstrap")
        rows = m.score(rows, units)
    evaluation = m.reduce([r for r in rows if r["split"] == "evaluation"])
    stress = {str(size): m.reduce([r for r in rows if r["size"] == size]) for size in (16, 32, 64)}
    m.progress("after_labels_and_bootstrap", len(rows), 0)
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_local" if pre["external_ready"] else "blocked_no_run",
        inference_substrate_class="cpu_exact_solver_or_simulator"
        if pre["external_ready"]
        else "blocked_no_run",
        inference_substrate_details=dict(scope="paid encoded exact requests; no model or training"),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(
                phase="authentication_complete_requests_and_bootstrap",
                start_s=0,
                end_s=(end - start) / 1e9,
            )
        ],
    )
    return dict(
        preconditions_checked=pre,
        rows=rows,
        evaluation=evaluation,
        stress=stress,
        private_unit_limit=limit,
        current_work_receipt=receipt,
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse retained node outcomes, actual child coverage and private E2E-018/021 checks."""
    validator = namespace(previous, "indexed8847_validation")
    validator.m = m
    return dict(validator.validate(raw, private))


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Readiness admits a complete honest study; gain remains a separate registered decision."""
    pre, rows, receipt = work["preconditions_checked"], work["rows"], work["current_work_receipt"]
    evaluation, stress = work["evaluation"], work["stress"]
    stress_quality = all(
        s["completed_decision_parity"]
        and all(
            s["acceptance_gates"][k]["passed"]
            for k in ("coverage_loss", "semantic_false_accept_increase")
        )
        for s in stress.values()
        if s["per_world_results"]
    )
    ready = int(
        pre["external_ready"]
        and pre["resources"]["passed"]
        and qualified(validation)
        and len(rows) == 19200
        and work["private_unit_limit"] is None
        and not any(r["status"] == "failed" for r in rows)
        and evaluation["completed_decision_parity"]
        and all(s["completed_decision_parity"] for s in stress.values())
    )
    benefit = int(ready and evaluation["indexed_cost_benefit_score"] and stress_quality)
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "circular_positive"
        if benefit
        else "null"
        if ready
        else "disqualified"
    )
    value = dict(
        experiment_id=8847,
        task_id=m.TASK,
        milestone="2026.11.757",
        run_date=date,
        honest_verdict=f"complete_blocked_{pre['diagnostics'][0]['check']}_indexed_cost"
        if verdict == "blocked"
        else f"complete_{verdict}_indexed_cost",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="exposed V756 encoded worlds and constructed stress; no language or hidden-game generalization",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        random_seed=m.SEED,
        phase_spans=receipt["phase_spans"],
        preconditions_checked=pre,
        source_artifact_hashes=pre["references"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"]
        + validation["raw_references"]
        + [work_ref, validation_ref],
        rows=rows,
        intended_count=19200,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len({r["cluster_id"] for r in rows}),
        gate_check_summary=pre["diagnostics"]
        + [
            dict(
                check="owned_validation_failure",
                upstream=m.TASK,
                artifact_field=c.get("name", "owned_statement_coverage"),
                expected=True,
                observed=False,
                path=c.get("stdout_path"),
                sha256=c.get("stdout_sha256"),
            )
            for c in validation["receipts"] + [dict(passed=validation["coverage"]["complete"])]
            if not c["passed"]
        ],
        acceptance_gates=dict(
            evaluation=evaluation["acceptance_gates"],
            stress_no_quality_regression=stress_quality,
            full_roster=len(rows) == 19200,
            owned_validation=qualified(validation),
        ),
        required_checks_passed=qualified(validation),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        indexed_cost_ready_score=ready,
        indexed_cost_benefit_score=benefit,
        per_world_results=evaluation["per_world_results"],
        paired_intervals=evaluation["paired_intervals"],
        wall_speed_ratio=evaluation["wall_speed_ratio"],
        stress_results=stress,
        complete_request_rows=[
            dict(
                unit_id=r["unit_id"],
                arm=r["arm"],
                seed=r["seed"],
                wall_time_s=r["wall_time_s"],
                cpu_time_s=r["cpu_time_s"],
                status=r["status"],
            )
            for r in rows
            if r["status"] == "completed"
        ],
        censored_request_rows=[r for r in rows if r["status"] == "censored"],
        index_build_rows=[
            dict(
                unit_id=r["unit_id"],
                seed=r["seed"],
                operations=r["operation_counts"]["index_build"],
                status=r["status"],
            )
            for r in rows
            if r["arm"] == m.ARMS[1]
        ],
        operation_rows=[
            dict(
                unit_id=r["unit_id"],
                arm=r["arm"],
                seed=r["seed"],
                paid=r["primitive_operations"],
                categories=r["operation_counts"],
            )
            for r in rows
        ],
        operation_schema=dict(
            cap=m.CAP,
            categories=list(m.witness.meter.KINDS) + list(m.witness.EXTRA) + list(m.EXTRA),
            definition="One paid semantic primitive or transport byte; disposal reservations precede allocation. Counts are not time fractions. Complete request clocks include decode, solver, serialization and disposal reservations; resident process import/compilation excluded equally.",
        ),
        current_work_receipt=receipt,
        work_reference=work_ref,
        validation_reference=validation_ref,
        repository_health=work.get("repository_health"),
        methodology="Frozen original exact evaluators; five paired process schedules; all attempts retained; 10000 whole-world bootstrap draws; Bonferroni familywise 99% gain intervals. No strategy or threshold search; a null closes this unchanged mechanism.",
    )
    value["field_principles"] = {
        k: "Bind measured original requests, all attempts and immutable operands; constructed encoded truth remains circular evidence."
        for k in value
    }
    value["field_principles"].update(
        indexed_cost_ready_score="Complete authenticated study and owned checks admit evidence independently of benefit.",
        indexed_cost_benefit_score="Only the frozen evaluation gain bounds and stress quality gates admit benefit.",
        per_world_results="Collapse variants and five scheduling repeats within each independent world.",
        paired_intervals="10000 paired whole-world draws; three Bonferroni intervals jointly cover 99 percent.",
        wall_speed_ratio="Paired completed request clocks yield a separate 99-percent speed interval; counts imply no speedup.",
        complete_request_rows="Retain measured wall and CPU times for completed request boundaries.",
        censored_request_rows="All exhausted requests remain insufficient and retain unit denominators.",
        index_build_rows="Charge actual index construction for every indexed request, including exhausted requests.",
        operation_rows="Retain full primitive category receipts so all paid reductions can be recomputed.",
        stress_results="Report each constructed fact size separately; these worlds establish no natural-language generalization.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Re-execute sealed decisions and paid work; recompute gates from retained measured clocks."""
    try:
        value = json.loads(path.read_bytes())
        work, validation = load(value["work_reference"]), load(value["validation_reference"])
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if validate_current_work_receipt(work["current_work_receipt"]):
            return False
        checker = namespace(ordered.checker, "indexed8847_coverage_replay")
        checker.m = m
        if value["required_checks_passed"] and not checker.validation_evidence(validation):
            return False
        if work["rows"]:
            raw = Path(value["work_reference"]["path"]).parent
            with TemporaryDirectory(prefix="exp8847-replay-") as directory:
                current = preconditions(Path(directory))
                if not current["external_ready"] or load(
                    reference(Path(directory) / "frozen_inputs.json")
                ) != load(reference(raw / "frozen_inputs.json")):
                    return False
                units = load(reference(raw / "frozen_inputs.json"))["rows"][
                    : work["private_unit_limit"]
                ]
                sealed = load(reference(raw / "sealed_predictions.json"))
                if sealed["commitment"] != canonical_hash(sealed["rows"]) or m.witness.stable(
                    m.score(sealed["rows"], units)
                ) != m.witness.stable(work["rows"]):
                    return False
                observed = process_measure(Path(directory), units)
                if m.witness.stable(observed) != m.witness.stable(sealed["rows"]):
                    return False
            if (
                m.reduce([r for r in work["rows"] if r["split"] == "evaluation"])
                != work["evaluation"]
            ):
                return False
            if {
                str(s): m.reduce([r for r in work["rows"] if r["size"] == s]) for s in (16, 32, 64)
            } != work["stress"]:
                return False
        return bool(
            value
            == artifact(
                work,
                validation,
                value["run_date"],
                value["work_reference"],
                value["validation_reference"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Reuse unchanged scanners and the outside-checkout replay with PYTHONPATH removed."""
    checked = namespace(ordered, "indexed8847_terminal")
    checked.m = m
    return dict(checked.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Private subsets exercise the real boundary but cannot qualify public readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--private-unit-limit", type=int)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--measure-arm", choices=m.ARMS)
    parser.add_argument("--schedule-seed", type=int, choices=m.SCHEDULES)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args(argv)
    if args.measure_arm:
        if args.inputs is None or args.decisions is None or args.schedule_seed is None:
            parser.error("measurement worker requires input, output and registered scheduling seed")
        rows = m.measure(load(reference(args.inputs))["rows"], args.schedule_seed, args.measure_arm)
        atomic_json(args.decisions, dict(rows=rows))
        return 0
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private output must be outside checkout")
    if args.private_unit_limit is not None and (
        not args.private_output or args.private_unit_limit < 1
    ):
        parser.error("subset requires private output and positive count")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    try:
        work = prepare(raw, args.private_unit_limit)
    except (ValueError, KeyError, TypeError) as error:
        work = prepare(raw, args.private_unit_limit, False)
        work["preconditions_checked"]["resources"]["passed"] = False
        work["preconditions_checked"]["diagnostics"].append(
            dict(
                check="internal_failure",
                upstream=m.TASK,
                artifact_field="measurement",
                expected="complete owned execution",
                observed=str(error),
                path=str(raw),
                sha256=None,
            )
        )
    work["code_config_hashes"] = [
        reference(m.ROOT / p)
        for p in m.OWNED
        + m.TESTS
        + [
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
        ]
    ]
    work["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    if HEALTH.is_file():
        work["repository_health"] = load(reference(HEALTH))
        work["raw_shard_hashes"] += [reference(HEALTH)] + [
            reference(p) for p in sorted(HEALTH.parent.glob("full_suite.*"))
        ]
    atomic_json(raw / "work.json", work)
    with TemporaryDirectory(prefix="exp8847-owned-") as directory:
        m.progress("before_owned_validation")
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
        m.progress("after_owned_validation")
    atomic_json(raw / "validation.json", validation)
    value = artifact(
        work,
        validation,
        args.date,
        reference(raw / "work.json"),
        reference(raw / "validation.json"),
    )
    m.progress("before_publication")
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    m.progress("complete_" + value["verdict_class"], len(value["rows"]), 0)
    return 0
