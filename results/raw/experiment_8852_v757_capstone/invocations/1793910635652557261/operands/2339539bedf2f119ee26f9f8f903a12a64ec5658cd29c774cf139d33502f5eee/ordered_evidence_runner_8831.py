"""REQ-VERIFY-8831: bind raw paid evidence to checked terminal publication."""

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.reporting import ordered_evidence_cost_8831 as m
from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting import evidence_decisive_runner_8818 as checked
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

Json = dict[str, Any]
HEALTH = Path("/tmp/exp8831-repository-health/receipt.json")
checker = namespace(checked, "ordered8831_checks")
checker.m = m


def measurement_units(cohort: Json, private_world_limit: int | None) -> list[Json]:
    """Private smoke cohorts are explicit subsets and cannot qualify full-cohort readiness."""
    chosen = {
        w["world_id"]
        for split in ["calibration", "evaluation"]
        for w in [w for w in cohort["worlds"] if w["split"] == split][:private_world_limit]
    }
    return [u for u in cohort["inputs"] if u["cluster_id"] in chosen]


def preconditions(raw: Path) -> Json:
    """Verify external byte authority and local capacity before any benchmark starts."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(probe="private output"))
    refs, diagnostics = [], []
    diagnostic: Json = dict(
        check="missing_file",
        upstream="exp8829-selective-methods",
        artifact_field="evidence_protocol_ready_score",
        expected=1,
        observed=None,
        path=str(m.UPSTREAM),
        sha256=None,
    )
    try:
        ref = reference(m.UPSTREAM)
        value = load(ref)
        diagnostic.update(
            sha256=ref["sha256"],
            check="missing_field",
            observed=value.get("evidence_protocol_ready_score"),
        )
        if "evidence_protocol_ready_score" not in value:
            raise ValueError("missing_field")
        diagnostic["check"] = "observed_zero"
        if value["evidence_protocol_ready_score"] != 1:
            raise ValueError("readiness_zero")
        diagnostic.update(
            check="rejected_producer",
            artifact_field="sha256",
            expected=m.PIN,
            observed=ref["sha256"],
        )
        if (
            ref["sha256"] != m.PIN
            or value["task_id"] != "exp8829-selective-methods"
            or not value["required_checks_passed"]
        ):
            raise ValueError("producer_identity")
        refs.append(ref)
        for operand in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            diagnostic.update(
                path=operand["path"],
                artifact_field="sha256",
                expected=operand["sha256"],
                observed=None,
            )
            observed = sha256_file(Path(operand["path"]))
            diagnostic.update(observed=observed, sha256=observed)
            if observed != operand["sha256"]:
                raise ValueError("operand_drift")
            refs.append(operand)
        for name in ["cohort.json", "inputs.json", "protocol.json", "selection.json"]:
            operand = next(
                r
                for r in value["raw_shard_hashes"]
                if Path(r["path"]).name == name
                and Path(r["path"]).parent == Path(value["work_reference"]["path"]).parent
            )
            shutil.copyfile(operand["path"], raw / name)
        shutil.copyfile(m.UPSTREAM, raw / "upstream.json")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        diagnostics.append(
            dict(
                diagnostic,
                check="missing_file" if isinstance(error, OSError) else diagnostic["check"],
                error=str(error),
            )
        )
    resources: Json = dict(
        repository_virtualenv=(m.ROOT / ".venv/bin/python").is_file(),
        cpu_numerical_dependencies=np.__version__,
        private_output_writable=(raw / "writable.json").is_file(),
        free_disk_bytes=shutil.disk_usage(raw).free,
    )
    resources["passed"] = bool(
        resources["repository_virtualenv"] and resources["free_disk_bytes"] >= 128 * 1024 * 1024
    )
    return dict(
        resources=resources,
        references=refs,
        diagnostics=diagnostics,
        external_ready=not diagnostics,
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse scoped exact-node coverage and private authority/audit consumer checks."""
    validator = namespace(shared, "ordered8831_validation")
    validator.e = SimpleNamespace(OWNED=m.OWNED, TESTS=m.TESTS, ROOT=m.ROOT)
    validator.original = SimpleNamespace(
        CONSUMERS=[
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    validator.child = stream
    return dict(validator.validate(raw, private))


def prepare(raw: Path, private_world_limit: int | None = None, measurement: bool = True) -> Json:
    """Seal calibration choice and input-only evaluation before independently opening truth."""
    start = time.monotonic_ns()
    m.progress("before_preconditions")
    pre = preconditions(raw)
    m.progress("after_preconditions")
    decisions, rows, controls, references = [], [], [], []
    selected: Json = dict(schedule=None)
    result: Json = dict(
        per_world_results=[],
        whole_world_intervals={},
        acceptance_gates={},
        schedule_benefit_score=0,
    )
    if measurement and pre["external_ready"] and pre["resources"]["passed"]:
        cohort = load(reference(raw / "cohort.json"))
        protocol = load(reference(raw / "protocol.json"))
        producer = load(reference(raw / "upstream.json"))
        if (
            protocol["schedules"] != m.SCHEDULES
            or protocol["ordered_checks"] != m.base.CHECKS
            or protocol["primitive_cap"] != m.CAP
        ):
            raise ValueError("frozen_schedule_drift")
        worlds = cohort["worlds"]
        if len({w["structure"] for w in worlds}) != len(worlds) or any(
            producer["split_manifest"][s] != [w["world_id"] for w in worlds if w["split"] == s]
            for s in m.base.COUNTS
        ):
            raise ValueError("split_manifest_drift")
        inputs = measurement_units(cohort, private_world_limit)
        atomic_json(raw / "measurement_inputs.json", dict(rows=inputs))
        for split in ["calibration", "evaluation"]:
            units = [u for u in inputs if u["split"] == split]
            measured = m.measure(units)
            atomic_json(
                raw / (split + "_decisions.json"),
                dict(rows=measured, commitment=canonical_hash(measured)),
            )
            scored = m.score(measured, units)
            if split == "calibration":
                selected = m.select(scored)
                atomic_json(raw / "selected_schedule.json", selected)
            else:
                decisions, rows = measured, scored
                result = m.reduce(rows, selected)
        m.progress("before_unrestricted_reference", 0, len(decisions) // 3)
        references = [
            dict(
                m.attempt("always_joint", u["input"], None),
                unit_id=u["unit_id"],
                scope="unrestricted_reference_only",
            )
            for u in inputs
            if u["split"] == "evaluation"
        ]
        m.progress("after_unrestricted_reference", len(references), 0)
        controls = m.controls()
        atomic_json(raw / "controls.json", dict(rows=controls))
        atomic_json(raw / "reference_costs.json", dict(rows=references))
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_local",
        inference_substrate_details=dict(
            scope="exact encoded signed graphs; CPU operations; no training"
        ),
        inference_substrate_class="cpu_exact_solver_or_simulator",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(
                phase="preconditions_calibration_evaluation_reference",
                start_s=0,
                end_s=(end - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    return dict(
        private_world_limit=private_world_limit,
        preconditions_checked=pre,
        rows=rows,
        decision_rows=decisions,
        selected_schedule=selected,
        controls=controls,
        reference_costs=references,
        current_work_receipt=receipt,
        **result,
    )


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Separate valid cost evidence from a useful schedule and from external generalization."""
    pre, rows, receipt = work["preconditions_checked"], work["rows"], work["current_work_receipt"]
    valid = qualified(validation)
    intact = all(c["passed"] for c in work["controls"]) and len(rows) == 4320
    ready = int(
        valid
        and intact
        and pre["external_ready"]
        and pre["resources"]["passed"]
        and work["private_world_limit"] is None
    )
    benefit = ready * work["schedule_benefit_score"]
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if not ready
        else "circular_positive"
        if benefit
        else "null"
    )
    gates = dict(
        work["acceptance_gates"],
        evidence_readiness=dict(
            passed=bool(ready),
            principle="Owned checks, exact operands, complete scheduled denominator and controls qualify evidence independently of benefit",
        ),
    )
    value = dict(
        experiment_id=8831,
        task_id=m.TASK,
        milestone="2026.11.756",
        run_date=date,
        honest_verdict=f"complete_blocked_{pre['diagnostics'][0]['check']}_ordered_evidence_cost"
        if verdict == "blocked"
        else f"complete_{verdict}_ordered_evidence_cost",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="constructed encoded worlds; exact symbolic oracle; conditioned topology sampling",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        random_seed=m.SEED,
        source_artifact_hashes=pre["references"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"]
        + validation["raw_references"]
        + [work_ref, validation_ref],
        phase_spans=receipt["phase_spans"],
        preconditions_checked=pre,
        current_work_receipt=receipt,
        rows=rows,
        operation_rows=work["decision_rows"],
        decision_rows=work["decision_rows"],
        intended_count=4320,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len(work["per_world_results"]),
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=gates,
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        evidence_rows_ready_score=ready,
        schedule_benefit_score=benefit,
        per_world_results=work["per_world_results"],
        selected_schedule=work["selected_schedule"],
        whole_world_intervals=work["whole_world_intervals"],
        reference_costs=work["reference_costs"],
        secondary_wall_time=work.get("secondary_wall_time"),
        controls=work["controls"],
        work_reference=work_ref,
        validation_reference=validation_ref,
        methodology="Three frozen schedules; 200000 actual explicit operations per request including parsing, signs and dispatch; calibration selects matched-decision ordered cost; predictions sealed before independent completion-enumeration labels; 10000 paired whole-world bootstrap draws, four 98.75 percent intervals. Successful bindings count the first complete accepted witness, not an exhaustive witness census. Reference costs are outside request budgets. No model calls or training.",
        assumptions=[
            "Conditioned topology sampling supports finite-cohort comparison, not an iid generalization certificate.",
            "Explicit interpreter operations are not machine instructions; wall time is secondary.",
        ],
        repository_health=work.get("repository_health"),
    )
    value["field_principles"] = {
        k: (
            "Paired exact decision quality and lower paid work determine scientific benefit independently of execution readiness"
            if k in ["schedule_benefit_score", "whole_world_intervals", "acceptance_gates"]
            else "Retain measured unit outcomes, exact byte provenance and current producer identity so replay can reject changed evidence"
        )
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute decisions and reductions so repaired hashes cannot hide an invented result."""
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
        if value["required_checks_passed"] and not checker.validation_evidence(validation):
            return False
        raw = Path(value["work_reference"]["path"]).parent
        if work["rows"]:
            inputs = load(reference(raw / "measurement_inputs.json"))["rows"]
            cohort = load(reference(raw / "cohort.json"))
            upstream = load(reference(raw / "upstream.json"))
            expected_cohort = next(
                ref
                for ref in upstream["raw_shard_hashes"]
                if Path(ref["path"]).name == "cohort.json"
                and Path(ref["path"]).parent == Path(upstream["work_reference"]["path"]).parent
            )
            if (
                sha256_file(raw / "upstream.json") != m.PIN
                or sha256_file(raw / "cohort.json") != expected_cohort["sha256"]
                or inputs != measurement_units(cohort, work["private_world_limit"])
            ):
                return False
            if m.stable(m.reduce(work["rows"], work["selected_schedule"])) != m.stable(
                {
                    k: work[k]
                    for k in [
                        "per_world_results",
                        "whole_world_intervals",
                        "acceptance_gates",
                        "schedule_benefit_score",
                        "secondary_wall_time",
                    ]
                }
            ):
                return False
            for split in ["calibration", "evaluation"]:
                units = [u for u in inputs if u["split"] == split]
                sealed = load(reference(raw / (split + "_decisions.json")))
                measured = m.measure(units)
                if m.stable(measured) != m.stable(sealed["rows"]) or sealed[
                    "commitment"
                ] != canonical_hash(sealed["rows"]):
                    return False
                scored = m.score(sealed["rows"], units)
                if split == "calibration":
                    if m.select(scored) != work["selected_schedule"]:
                        return False
                elif scored != work["rows"] or m.stable(
                    m.reduce(scored, work["selected_schedule"])
                ) != m.stable(
                    {
                        k: work[k]
                        for k in [
                            "per_world_results",
                            "whole_world_intervals",
                            "acceptance_gates",
                            "schedule_benefit_score",
                            "secondary_wall_time",
                        ]
                    }
                ):
                    return False
            if m.stable(m.controls()) != m.stable(work["controls"]):
                return False
            units = [u for u in inputs if u["split"] == "evaluation"]
            for unit, row in zip(units, work["reference_costs"], strict=True):
                if m.stable(m.attempt("always_joint", unit["input"], None)) != m.stable(
                    {k: v for k, v in row.items() if k not in ["unit_id", "scope"]}
                ):
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
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Unchanged adversarial and row scanners inspect the exact prospective primary bytes."""
    checks = []
    for name, command in [
        ("adversarial", [".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(path)]),
        (
            "row_consistency",
            [
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(path),
            ],
        ),
    ]:
        m.progress("before_" + name)
        checks.append(stream(name, command, raw / "terminal_checks", deadline=120))
        m.progress("after_" + name)
    with TemporaryDirectory(prefix="exp8831-cold-") as directory:
        m.progress("before_outside_checkout_replay")
        command = [
            "env",
            "-u",
            "PYTHONPATH",
            str(m.ROOT / ".venv/bin/python"),
            "-u",
            "-c",
            "import os,runpy,sys; os.chdir(sys.argv.pop(1)); sys.argv[0]=sys.argv.pop(1); runpy.run_path(sys.argv[0],run_name='__main__')",
            directory,
            str(m.ROOT / m.CLI),
            "--cold-replay",
            str(path),
        ]
        checks.append(stream("cold_replay", command, raw / "terminal_checks", deadline=600))
        m.progress("after_outside_checkout_replay")
    return dict(passed=all(c["passed"] for c in checks), receipts=checks)


def main(argv: list[str] | None = None) -> int:
    """Private runs exercise real science while requiring independent owned qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--private-world-limit", type=int)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    if args.private_world_limit is not None and (
        not args.private_output or args.private_world_limit < 1
    ):
        parser.error("world limit requires private output and a positive count")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    try:
        work = prepare(raw, args.private_world_limit)
    except (ValueError, KeyError, TypeError) as error:
        work = prepare(raw, args.private_world_limit, False)
        work["preconditions_checked"]["resources"]["passed"] = False
        work["preconditions_checked"]["diagnostics"].append(
            dict(
                check="internal_failure",
                upstream=m.TASK,
                artifact_field="measurement",
                expected="complete exact execution",
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
            "python/carnot/reporting/evidence_operation_meter_8818.py",
            "python/carnot/reporting/selective_methods_8829.py",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
        ]
    ]
    work["raw_shard_hashes"] = [reference(p) for p in sorted(raw.glob("*.json"))]
    if HEALTH.is_file():
        health = json.loads(HEALTH.read_bytes())
        for suffix in ["stdout", "stderr"]:
            target = raw / ("repository_health." + suffix)
            shutil.copyfile(health[suffix + "_path"], target)
            work["raw_shard_hashes"].append(reference(target))
        atomic_json(
            raw / "repository_health.json",
            dict(receipt=health, scope="repository_health_debt", affects_owned_readiness=False),
        )
        work["repository_health"] = reference(raw / "repository_health.json")
        work["raw_shard_hashes"].append(work["repository_health"])
    atomic_json(raw / "work.json", work)
    with TemporaryDirectory(prefix="exp8831-owned-") as directory:
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
